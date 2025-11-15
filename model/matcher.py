"""Hungarian Matcher for DETR-style models."""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple


class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher for matching predictions to ground truth.
    
    For single-object detection, this is simplified since we always have
    exactly 1 ground truth object per image.
    """
    
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        """
        Args:
            cost_class: Weight for classification cost
            cost_bbox: Weight for L1 bbox cost
            cost_giou: Weight for GIoU cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    @torch.no_grad()
    def forward(self, outputs: dict, targets: List[dict]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform matching.
        
        Args:
            outputs: Dict with 'pred_logits' (B, num_queries, 1) and 'pred_boxes' (B, num_queries, 4)
            targets: List of dicts, each with 'labels' (1,) and 'boxes' (1, 4)
            
        Returns:
            indices: List of (pred_idx, tgt_idx) tuples for each batch element
        """
        B, num_queries = outputs["pred_logits"].shape[:2]
        
        # Flatten batch dimension
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # (B*num_queries, 1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # (B*num_queries, 4)
        
        # Concatenate target boxes
        tgt_bbox = torch.cat([t["boxes"] for t in targets])  # (B, 4) - one box per image
        
        # Compute classification cost (binary)
        # Since we have single object, target is always 1
        cost_class = -out_prob.squeeze(-1)  # (B*num_queries,) - higher prob = lower cost
        
        # Compute L1 bbox cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # (B*num_queries, B)
        
        # Compute GIoU cost
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)  # (B*num_queries, B)
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class.unsqueeze(1) + self.cost_giou * cost_giou
        C = C.view(B, num_queries, -1).cpu()
        
        # Perform matching for each batch element
        sizes = [1 for _ in range(B)]  # Each image has 1 target
        indices = []
        for i, (c, size) in enumerate(zip(C.split(sizes, -1), sizes)):
            # c: (num_queries, 1)
            c = c[i].squeeze(-1)  # (num_queries,)
            pred_idx = c.argmin().unsqueeze(0)  # Select query with minimum cost
            tgt_idx = torch.tensor([0], dtype=torch.int64)  # Always match to the single target
            indices.append((pred_idx, tgt_idx))
        
        return indices


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format."""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """Convert boxes from [x1, y1, x2, y2] to [cx, cy, w, h] format."""
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: (N, 4) in [x1, y1, x2, y2] format
        boxes2: (M, 4) in [x1, y1, x2, y2] format
        
    Returns:
        iou: (N, M) IoU matrix
        union: (N, M) Union area matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
    
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
    
    union = area1[:, None] + area2 - inter
    
    iou = inter / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute Generalized IoU between two sets of boxes.
    
    Args:
        boxes1: (N, 4) in [cx, cy, w, h] format
        boxes2: (M, 4) in [cx, cy, w, h] format
        
    Returns:
        giou: (N, M) GIoU matrix
    """
    # Convert to xyxy format
    boxes1_xyxy = box_cxcywh_to_xyxy(boxes1)
    boxes2_xyxy = box_cxcywh_to_xyxy(boxes2)
    
    # Compute IoU
    iou, union = box_iou(boxes1_xyxy, boxes2_xyxy)
    
    # Compute enclosing box
    lt = torch.min(boxes1_xyxy[:, None, :2], boxes2_xyxy[:, :2])
    rb = torch.max(boxes1_xyxy[:, None, 2:], boxes2_xyxy[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    area_c = wh[:, :, 0] * wh[:, :, 1]
    
    # GIoU = IoU - (area_c - union) / area_c
    giou = iou - (area_c - union) / (area_c + 1e-6)
    
    return giou


if __name__ == "__main__":
    # Test matcher
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dummy outputs
    outputs = {
        "pred_logits": torch.randn(2, 5, 1).to(device),  # 2 images, 5 queries
        "pred_boxes": torch.rand(2, 5, 4).to(device)     # Random boxes in [0, 1]
    }
    
    # Create dummy targets (1 object per image)
    targets = [
        {"labels": torch.tensor([1], device=device), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]], device=device)},
        {"labels": torch.tensor([1], device=device), "boxes": torch.tensor([[0.3, 0.4, 0.15, 0.25]], device=device)}
    ]
    
    matcher = HungarianMatcher().to(device)
    indices = matcher(outputs, targets)
    
    print("Matching results:")
    for i, (pred_idx, tgt_idx) in enumerate(indices):
        print(f"  Image {i}: Query {pred_idx.item()} matched to target {tgt_idx.item()}")

