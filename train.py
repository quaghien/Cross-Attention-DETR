"""Training script for CA-DETR."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import build_ca_detr
from model.losses import build_criterion
from utils.dataset import build_dataset, collate_fn


def get_lr_scheduler(optimizer, args, steps_per_epoch):
    """Build learning rate scheduler."""
    if args.lr_schedule == 'cosine':
        total_steps = args.epochs * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=args.min_lr
        )
    elif args.lr_schedule == 'linear':
        def lr_lambda(step):
            total_steps = args.epochs * steps_per_epoch
            if step < total_steps:
                return max(args.min_lr / args.lr, 1.0 - step / total_steps)
            return args.min_lr / args.lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:  # constant
        scheduler = None
    
    return scheduler


def train_one_epoch(model, criterion, loader, optimizer, scheduler, device, epoch: int) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_loss_ce = 0.0
    total_loss_bbox = 0.0
    total_loss_giou = 0.0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", ncols=120)
    for templates, searches, targets in pbar:
        templates = templates.to(device)
        searches = searches.to(device)
        
        # Move targets to device
        for t in targets:
            t['boxes'] = t['boxes'].to(device)
            t['labels'] = t['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred_logits, pred_boxes = model(templates, searches)
        
        # Prepare outputs dict
        outputs = {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes
        }
        
        # Compute losses
        losses = criterion(outputs, targets)
        
        # Total loss
        loss = sum(losses.values())
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_loss_ce += losses.get('loss_ce', 0).item()
        total_loss_bbox += losses.get('loss_bbox', 0).item()
        total_loss_giou += losses.get('loss_giou', 0).item()
        
        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "ce": f"{losses.get('loss_ce', 0).item():.4f}",
            "bbox": f"{losses.get('loss_bbox', 0).item():.4f}",
            "giou": f"{losses.get('loss_giou', 0).item():.4f}",
            "lr": f"{current_lr:.2e}"
        })
    
    steps = len(loader)
    return {
        "loss": total_loss / steps,
        "loss_ce": total_loss_ce / steps,
        "loss_bbox": total_loss_bbox / steps,
        "loss_giou": total_loss_giou / steps,
    }


def evaluate(model, criterion, loader, device, epoch: int) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    total_loss_ce = 0.0
    total_loss_bbox = 0.0
    total_loss_giou = 0.0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]", ncols=120)
        for templates, searches, targets in pbar:
            templates = templates.to(device)
            searches = searches.to(device)
            
            # Move targets to device
            for t in targets:
                t['boxes'] = t['boxes'].to(device)
                t['labels'] = t['labels'].to(device)
            
            # Forward pass
            pred_logits, pred_boxes = model(templates, searches)
            
            # Prepare outputs dict
            outputs = {
                'pred_logits': pred_logits,
                'pred_boxes': pred_boxes
            }
            
            # Compute losses
            losses = criterion(outputs, targets)
            
            # Total loss
            loss = sum(losses.values())
            
            # Accumulate losses
            total_loss += loss.item()
            total_loss_ce += losses.get('loss_ce', 0).item()
            total_loss_bbox += losses.get('loss_bbox', 0).item()
            total_loss_giou += losses.get('loss_giou', 0).item()
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ce": f"{losses.get('loss_ce', 0).item():.4f}",
                "bbox": f"{losses.get('loss_bbox', 0).item():.4f}",
                "giou": f"{losses.get('loss_giou', 0).item():.4f}"
            })
    
    steps = len(loader)
    return {
        "loss": total_loss / steps,
        "loss_ce": total_loss_ce / steps,
        "loss_bbox": total_loss_bbox / steps,
        "loss_giou": total_loss_giou / steps,
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
        pretrained_backbone=True
    )
    model.to(device)
    
    num_params = model.get_num_params()
    print(f"Model parameters: {num_params/1e6:.2f}M")
    
    # Build criterion
    weight_dict = {
        'loss_ce': args.loss_ce_weight,
        'loss_bbox': args.loss_bbox_weight,
        'loss_giou': args.loss_giou_weight
    }
    criterion = build_criterion(
        num_classes=1,
        weight_dict=weight_dict,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma
    )
    criterion.to(device)
    
    # Build datasets
    print(f"\nLoading datasets from: {args.data_dir}")
    train_dataset = build_dataset(
        root=args.data_dir,
        split="train",
        augment=True,
        augment_prob=args.augment_prob,
        img_size=args.img_size
    )
    val_dataset = build_dataset(
        root=args.data_dir,
        split="val",
        augment=False,
        img_size=args.img_size
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Build dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Build optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Build scheduler
    scheduler = get_lr_scheduler(optimizer, args, len(train_loader))
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.checkpoint_path and Path(args.checkpoint_path).exists():
        print(f"\nLoading checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs")
    history = []
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_one_epoch(model, criterion, train_loader, optimizer, scheduler, device, epoch)
        
        # Validate
        val_metrics = evaluate(model, criterion, val_loader, device, epoch)
        
        # Log metrics
        metrics = {
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics
        }
        history.append(metrics)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, CE: {train_metrics['loss_ce']:.4f}, "
              f"BBox: {train_metrics['loss_bbox']:.4f}, GIoU: {train_metrics['loss_giou']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, CE: {val_metrics['loss_ce']:.4f}, "
              f"BBox: {val_metrics['loss_bbox']:.4f}, GIoU: {val_metrics['loss_giou']:.4f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, output_dir / "best_model.pth")
            print(f"  → Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Save last model
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, output_dir / f"last_model_epoch_{epoch+1}.pth")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch+1}.pth")
            print(f"  → Saved checkpoint at epoch {epoch+1}")
        
        # Save history
        with open(output_dir / "history.json", 'w') as f:
            json.dump(history, f, indent=2)
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CA-DETR")
    
    # Data
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--img_size", type=int, default=640, help="Image size")
    
    # Model
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "efficientnet_b0"],
                       help="Backbone architecture")
    parser.add_argument("--num_queries", type=int, default=1, help="Number of object queries")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num_encoder_layers", type=int, default=3, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=3, help="Number of decoder layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dim_feedforward", type=int, default=1024, help="FFN dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Loss
    parser.add_argument("--loss_ce_weight", type=float, default=1.0, help="Classification loss weight")
    parser.add_argument("--loss_bbox_weight", type=float, default=5.0, help="BBox L1 loss weight")
    parser.add_argument("--loss_giou_weight", type=float, default=2.0, help="GIoU loss weight")
    parser.add_argument("--focal_alpha", type=float, default=0.25, help="Focal loss alpha")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--lr_schedule", type=str, default="cosine", choices=["constant", "cosine", "linear"],
                       help="Learning rate schedule")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--augment_prob", type=float, default=0.75, help="Augmentation probability")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers")
    
    # Checkpointing
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    
    main(args)

