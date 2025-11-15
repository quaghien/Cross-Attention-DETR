"""Inference script for CA-DETR."""

import argparse
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import cv2
import torch
from tqdm import tqdm

from model import build_ca_detr
from utils.transforms import build_transforms


def extract_frame_number(filename: str) -> int:
    """Extract frame number from filename."""
    name = Path(filename).stem
    parts = name.split('_')
    
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    
    digits = ''
    for char in reversed(name):
        if char.isdigit():
            digits = char + digits
        elif digits:
            break
    
    if digits:
        return int(digits)
    
    return 0


def convert_bbox_to_pixel(bbox_norm: torch.Tensor, img_w: int, img_h: int) -> Dict[str, int]:
    """
    Convert normalized bbox [cx, cy, w, h] to pixel coordinates [x1, y1, x2, y2].
    
    Args:
        bbox_norm: (4,) - [cx, cy, w, h] in normalized coordinates (0-1)
        img_w: Image width in pixels
        img_h: Image height in pixels
        
    Returns:
        bbox: Dict with x1, y1, x2, y2 in pixel coordinates
    """
    cx, cy, w, h = bbox_norm
    
    # Convert to pixel coordinates
    cx_pixel = cx * img_w
    cy_pixel = cy * img_h
    w_pixel = w * img_w
    h_pixel = h * img_h
    
    # Convert center to corner coordinates
    x1 = int(cx_pixel - w_pixel / 2)
    y1 = int(cy_pixel - h_pixel / 2)
    x2 = int(cx_pixel + w_pixel / 2)
    y2 = int(cy_pixel + h_pixel / 2)
    
    # Clamp to image bounds
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))
    
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def extract_video_id(filename: str) -> str:
    """Extract video_id from filename."""
    name = Path(filename).stem
    parts = name.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return parts[0]


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
    
    # Setup transform (no augmentation for inference)
    transform = build_transforms(img_size=args.img_size, augment=False)
    
    # Collect all videos and their frames
    data_dir = Path(args.data_dir)
    search_images_dir = data_dir / args.split / "search" / "images"
    template_dir = data_dir / args.split / "templates"
    
    # Group frames by video_id
    video_frames = defaultdict(list)
    image_files = sorted(search_images_dir.glob("*.jpg")) + sorted(search_images_dir.glob("*.png"))
    
    for img_path in image_files:
        video_id = extract_video_id(img_path.name)
        frame_num = extract_frame_number(img_path.name)
        video_frames[video_id].append((frame_num, img_path))
    
    # Sort frames by frame number for each video
    for video_id in video_frames:
        video_frames[video_id].sort(key=lambda x: x[0])
    
    # Get templates for each video
    video_templates = defaultdict(list)
    template_files = list(template_dir.glob("*.jpg")) + list(template_dir.glob("*.png"))
    for template_path in template_files:
        video_id = extract_video_id(template_path.name)
        video_templates[video_id].append(template_path)
    
    print(f"Found {len(video_frames)} videos")
    
    # Process each video
    results = []
    
    with torch.no_grad():
        for video_id in tqdm(sorted(video_frames.keys()), desc="Processing videos"):
            frames = video_frames[video_id]
            templates = video_templates.get(video_id, [])
            
            if not templates:
                # No template found, add empty detections
                results.append({
                    "video_id": video_id,
                    "detections": []
                })
                continue
            
            # Use first template
            template_path = templates[0]
            
            # Process all frames for this video
            detections = []
            bboxes_list = []
            
            for frame_num, img_path in frames:
                # Get original image dimensions for bbox conversion
                img = cv2.imread(str(img_path))
                if img is not None:
                    orig_h, orig_w = img.shape[:2]
                else:
                    orig_w, orig_h = args.img_size, args.img_size
                
                # Load and transform images
                template_img = transform(template_path, aug_params=None)
                search_img = transform(img_path, aug_params=None)
                
                # Add batch dimension
                template_batch = template_img.unsqueeze(0).to(device)
                search_batch = search_img.unsqueeze(0).to(device)
                
                # Forward pass
                pred_logits, pred_boxes = model(template_batch, search_batch)
                
                # Get predictions (single query)
                confidence = pred_logits[0, 0, 0].sigmoid().item()
                bbox_norm = pred_boxes[0, 0]  # (4,) - [cx, cy, w, h]
                
                # Filter by confidence threshold
                if confidence >= args.confidence_threshold:
                    # Convert to pixel coordinates
                    bbox_pixel = convert_bbox_to_pixel(bbox_norm, orig_w, orig_h)
                    
                    bboxes_list.append({
                        "frame": frame_num,
                        **bbox_pixel
                    })
            
            # Group detections (for now, all bboxes in one detection group)
            if bboxes_list:
                detections.append({"bboxes": bboxes_list})
            
            results.append({
                "video_id": video_id,
                "detections": detections
            })
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "submission.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nInference complete! Results saved to {output_path}")
    print(f"Total videos: {len(results)}")
    videos_with_detections = sum(1 for r in results if r["detections"])
    print(f"Videos with detections: {videos_with_detections}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for CA-DETR")
    
    # Data
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root directory")
    parser.add_argument("--split", type=str, default="public_test", help="Dataset split (default: public_test)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory (will save submission.json here)")
    parser.add_argument("--img_size", type=int, default=640, help="Image size")
    
    # Model (must match training config)
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "efficientnet_b0"],
                       help="Backbone architecture")
    parser.add_argument("--num_queries", type=int, default=1, help="Number of object queries")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num_encoder_layers", type=int, default=3, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=3, help="Number of decoder layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dim_feedforward", type=int, default=1024, help="FFN dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Inference
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Confidence threshold for detections")
    
    args = parser.parse_args()
    
    main(args)

