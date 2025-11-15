"""Evaluate model and compute IoU on validation set."""

import argparse
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from model import build_ca_detr
from utils.dataset import ReferenceDetectionDataset, collate_fn
from utils.transforms import build_transforms
from torch.utils.data import DataLoader


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format."""
    x_c, y_c, w, h = boxes.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: (N, 4) in [cx, cy, w, h] format (normalized 0-1)
        boxes2: (M, 4) in [cx, cy, w, h] format (normalized 0-1)
        
    Returns:
        iou: (N, M) IoU matrix
    """
    # Convert to xyxy format
    boxes1_xyxy = box_cxcywh_to_xyxy(boxes1)
    boxes2_xyxy = box_cxcywh_to_xyxy(boxes2)
    
    # Compute intersection
    lt = torch.max(boxes1_xyxy[:, None, :2], boxes2_xyxy[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1_xyxy[:, None, 2:], boxes2_xyxy[:, 2:])  # (N, M, 2)
    
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
    
    # Compute areas
    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])
    
    # Compute union
    union = area1[:, None] + area2 - inter
    
    iou = inter / (union + 1e-6)
    return iou


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build model
    print(f"\nBuilding model: {args.backbone}")
    model = build_ca_detr(
        backbone_name=args.backbone,
        num_queries=args.num_queries,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pretrained_backbone=False
    )
    model.to(device)
    model.eval()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    print("Model loaded successfully")
    
    # Load validation dataset
    print(f"\nLoading validation dataset from {args.data_dir}")
    val_dataset = ReferenceDetectionDataset(
        root=args.data_dir,
        split="val",
        augment=False,
        img_size=args.img_size
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    print(f"Validation samples: {len(val_dataset)}")
    
    # Evaluate
    all_ious = []
    num_samples = 0
    debug_count = 0
    
    with torch.no_grad():
        for templates, searches, targets in tqdm(val_loader, desc="Evaluating"):
            templates = templates.to(device)
            searches = searches.to(device)
            
            # Forward pass
            pred_logits, pred_boxes = model(templates, searches)
            
            # Process each sample in batch
            batch_size = templates.shape[0]
            for i in range(batch_size):
                # Get prediction (use first query, highest confidence)
                pred_bbox = pred_boxes[i, 0]  # (4,) - [cx, cy, w, h]
                pred_conf = pred_logits[i, 0, 0].sigmoid().item()
                
                # Get ground truth
                gt_bbox = targets[i]['boxes'][0]  # (4,) - [cx, cy, w, h]
                
                # Debug first few samples
                if debug_count < 3:
                    print(f"\n[Debug Sample {debug_count}]")
                    print(f"  Pred bbox: {pred_bbox.cpu().numpy()}")
                    print(f"  GT bbox: {gt_bbox.cpu().numpy()}")
                    print(f"  Pred conf: {pred_conf:.4f}")
                    debug_count += 1
                
                # Always compute IoU (don't filter by confidence for evaluation)
                # This gives us true model performance
                pred_bbox_tensor = pred_bbox.unsqueeze(0).to(device)  # (1, 4)
                gt_bbox_tensor = gt_bbox.unsqueeze(0).to(device)  # (1, 4)
                
                iou_matrix = compute_iou(pred_bbox_tensor, gt_bbox_tensor)
                iou = iou_matrix[0, 0].item()
                all_ious.append(iou)
                
                if debug_count <= 3:
                    print(f"  IoU: {iou:.4f}")
                
                num_samples += 1
    
    # Compute statistics
    if len(all_ious) > 0:
        mean_iou = sum(all_ious) / len(all_ious)
        print(f"\n{'='*60}")
        print(f"Evaluation Results:")
        print(f"{'='*60}")
        print(f"Total samples: {num_samples}")
        print(f"Mean IoU: {mean_iou:.4f}")
        print(f"Min IoU: {min(all_ious):.4f}")
        print(f"Max IoU: {max(all_ious):.4f}")
        
        # IoU distribution
        iou_ranges = [
            (0.0, 0.3, "Poor (0.0-0.3)"),
            (0.3, 0.5, "Fair (0.3-0.5)"),
            (0.5, 0.7, "Good (0.5-0.7)"),
            (0.7, 0.9, "Very Good (0.7-0.9)"),
            (0.9, 1.0, "Excellent (0.9-1.0)")
        ]
        
        print(f"\nIoU Distribution:")
        for low, high, label in iou_ranges:
            count = sum(1 for iou in all_ious if low <= iou < high)
            if high == 1.0:
                count = sum(1 for iou in all_ious if low <= iou <= high)
            percentage = (count / len(all_ious)) * 100
            print(f"  {label}: {count:4d} ({percentage:5.2f}%)")
        
        print(f"{'='*60}")
    else:
        print("\nNo detections found!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CA-DETR and compute IoU")
    
    # Data
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root directory")
    parser.add_argument("--img_size", type=int, default=640, help="Image size")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    
    # Model (must match training config)
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "efficientnet_b0"],
                       help="Backbone architecture")
    parser.add_argument("--num_queries", type=int, default=5, help="Number of object queries")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num_encoder_layers", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=6, help="Number of decoder layers")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="FFN dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Inference
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Confidence threshold for detections")
    
    args = parser.parse_args()
    
    main(args)

