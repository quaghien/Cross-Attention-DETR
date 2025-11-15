"""Dataset for CA-DETR reference-based detection."""

import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from .transforms import build_transforms


class ReferenceDetectionDataset(Dataset):
    """
    Dataset for reference-based object detection.
    
    Each sample contains:
    - template: Reference image of the object
    - search: Search image where object should be detected
    - bbox: Ground truth bounding box in [cx, cy, w, h] format (normalized 0-1)
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        augment: bool = True,
        augment_prob: float = 0.75,
        img_size: int = 640
    ):
        """
        Args:
            root: Root directory of dataset
            split: Dataset split ('train', 'val', 'test')
            augment: Enable augmentation
            augment_prob: Probability of applying augmentation
            img_size: Image size (square)
        """
        self.root = Path(root)
        self.split = split
        self.augment = augment and split == "train"
        self.augment_prob = augment_prob
        self.img_size = img_size
        
        # Paths
        self.template_dir = self.root / split / "templates"
        self.search_images_dir = self.root / split / "search" / "images"
        self.search_labels_dir = self.root / split / "search" / "labels"
        
        # Collect data
        self.template_paths = self._collect_templates()
        self.samples = self._collect_samples()
        
        # Build transforms
        self.transform = build_transforms(img_size=img_size, augment=self.augment)
    
    @staticmethod
    def _extract_video_id(filename: str) -> str:
        """Extract video_id from filename."""
        name = Path(filename).stem
        parts = name.split('_')
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return parts[0]
    
    def _collect_templates(self) -> Dict[str, List[Path]]:
        """Collect templates grouped by video_id."""
        templates = {}
        template_files = list(self.template_dir.glob("*.jpg")) + list(self.template_dir.glob("*.png"))
        
        for template_path in template_files:
            video_id = self._extract_video_id(template_path.name)
            if video_id not in templates:
                templates[video_id] = []
            templates[video_id].append(template_path)
        
        if not templates:
            raise RuntimeError(f"No templates found in {self.template_dir}")
        return templates
    
    def _collect_samples(self) -> List[Tuple[str, Path, Path]]:
        """Collect samples (video_id, image_path, label_path)."""
        samples = []
        image_files = sorted(self.search_images_dir.glob("*.jpg")) + sorted(self.search_images_dir.glob("*.png"))
        
        for img_path in image_files:
            label_path = self.search_labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                video_id = self._extract_video_id(img_path.name)
                if video_id in self.template_paths:
                    samples.append((video_id, img_path, label_path))
        
        if not samples:
            raise RuntimeError(f"No samples found in {self.search_images_dir}")
        
        return samples
    
    def _parse_label(self, label_path: Path) -> Tuple[float, float, float, float]:
        """
        Parse label file to get bbox coordinates.
        
        Format: class_id x_center y_center width height (normalized 0-1)
        
        Returns:
            (x_c, y_c, w, h): Normalized bbox coordinates
        """
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            if not line:
                raise ValueError(f"Empty label file: {label_path}")
            
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"Invalid label format in {label_path}: {line}")
            
            # Parse: class_id x_c y_c w h
            x_c, y_c, w, h = map(float, parts[1:5])
            
            # Clamp to valid range
            x_c = max(0.0, min(1.0, x_c))
            y_c = max(0.0, min(1.0, y_c))
            w = max(1e-6, min(1.0, w))
            h = max(1e-6, min(1.0, h))
            
            return x_c, y_c, w, h
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            Dict with:
            - template: (3, H, W) - Template image tensor
            - search: (3, H, W) - Search image tensor
            - bbox: (4,) - Ground truth bbox [cx, cy, w, h] (normalized)
            - label: (1,) - Class label (always 1 for single-object detection)
        """
        video_id, img_path, label_path = self.samples[idx]
        
        # Get template (random if multiple available)
        template_path = random.choice(self.template_paths[video_id])
        
        # Parse label
        x_c, y_c, w, h = self._parse_label(label_path)
        
        # Decide whether to augment
        should_augment = self.augment and random.random() < self.augment_prob
        
        # Generate shared augmentation parameters for template and search
        aug_params = None
        if should_augment:
            aug_params = {
                'angle': random.uniform(-5, 5),
                'flip_h': random.random() < 0.5,
                'flip_v': random.random() < 0.3,
                'brightness': random.uniform(0.7, 1.3),
                'contrast': random.uniform(0.8, 1.2),
                'saturation': random.uniform(0.8, 1.2),
            }
        
        # Transform images (with same geometric augmentation)
        template_tensor = self.transform(template_path, aug_params=aug_params)
        search_tensor = self.transform(img_path, aug_params=aug_params)
        
        # Note: For DETR-style models, bbox augmentation is typically handled
        # by the model/loss (since we use normalized coords and simple augmentations)
        # For more complex augmentations, you'd need to transform bbox coords too
        
        return {
            'template': template_tensor,
            'search': search_tensor,
            'bbox': torch.tensor([x_c, y_c, w, h], dtype=torch.float32),
            'label': torch.tensor([1], dtype=torch.int64)  # Binary: object present
        }


def collate_fn(batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """
    Custom collate function for DataLoader.
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        templates: (B, 3, H, W) - Batched template images
        searches: (B, 3, H, W) - Batched search images
        targets: List of dicts with 'boxes' (1, 4) and 'labels' (1,)
    """
    templates = torch.stack([item['template'] for item in batch])
    searches = torch.stack([item['search'] for item in batch])
    
    targets = []
    for item in batch:
        targets.append({
            'boxes': item['bbox'].unsqueeze(0),  # (1, 4)
            'labels': item['label']  # (1,)
        })
    
    return templates, searches, targets


def build_dataset(
    root: str,
    split: str = "train",
    augment: bool = True,
    augment_prob: float = 0.75,
    img_size: int = 640
) -> ReferenceDetectionDataset:
    """
    Build dataset.
    
    Args:
        root: Root directory
        split: Dataset split
        augment: Enable augmentation
        augment_prob: Augmentation probability
        img_size: Image size
        
    Returns:
        dataset: ReferenceDetectionDataset instance
    """
    return ReferenceDetectionDataset(
        root=root,
        split=split,
        augment=augment,
        augment_prob=augment_prob,
        img_size=img_size
    )


if __name__ == "__main__":
    # Test dataset
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset.py <data_dir>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    
    dataset = build_dataset(root=data_dir, split="train", augment=True)
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Template shape: {sample['template'].shape}")
    print(f"Search shape: {sample['search'].shape}")
    print(f"Bbox: {sample['bbox']}")
    print(f"Label: {sample['label']}")
    
    # Test collate_fn
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    templates, searches, targets = next(iter(loader))
    print(f"\nBatch:")
    print(f"  Templates: {templates.shape}")
    print(f"  Searches: {searches.shape}")
    print(f"  Targets: {len(targets)} samples")
    print(f"  First target boxes: {targets[0]['boxes'].shape}")

