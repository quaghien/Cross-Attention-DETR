"""Image transformation utilities for CA-DETR."""

import random
from typing import Callable, Union, Tuple
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def _load_image(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Load image from path or return if already numpy array."""
    if isinstance(image, np.ndarray):
        return image
    image = cv2.imread(str(image))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image}")
    return image


def build_transforms(img_size: int = 640, augment: bool = True) -> Callable:
    """
    Build image transformation pipeline.
    
    Pipeline: Load → Resize → [Augmentation] → Normalize
    
    Args:
        img_size: Target square size (default 640)
        augment: Enable augmentation (default True)
        
    Returns:
        Callable that returns normalized tensor (3, img_size, img_size)
    """
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def transform(image: Union[str, Path, np.ndarray], aug_params: dict = None) -> torch.Tensor:
        """
        Transform image to normalized tensor.
        
        Args:
            image: Input image (path or numpy array)
            aug_params: Augmentation parameters (for syncing template/search)
            
        Returns:
            Normalized tensor (3, img_size, img_size)
        """
        # Load + convert BGR→RGB + resize
        img = _load_image(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        pil = T.functional.to_pil_image(img)
        tensor = T.functional.to_tensor(pil)
        
        if augment:
            if aug_params is None:
                # Generate random augmentation parameters
                aug_params = {
                    'angle': random.uniform(-5, 5),           # Rotation ±5°
                    'flip_h': random.random() < 0.5,          # H-flip 50%
                    'flip_v': random.random() < 0.3,          # V-flip 30%
                    'brightness': random.uniform(0.7, 1.3),   # ±30%
                    'contrast': random.uniform(0.8, 1.2),     # ±20%
                    'saturation': random.uniform(0.8, 1.2),   # ±20%
                }
            
            # Apply geometric augmentation
            if abs(aug_params['angle']) > 0.1:
                tensor = TF.rotate(tensor, aug_params['angle'], interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            
            if aug_params['flip_h']:
                tensor = TF.hflip(tensor)
            
            if aug_params['flip_v']:
                tensor = TF.vflip(tensor)
            
            # Apply color augmentation
            pil = T.functional.to_pil_image(tensor)
            pil = TF.adjust_brightness(pil, aug_params['brightness'])
            pil = TF.adjust_contrast(pil, aug_params['contrast'])
            pil = TF.adjust_saturation(pil, aug_params['saturation'])
            tensor = T.functional.to_tensor(pil)
        
        # Normalize (ImageNet stats)
        return normalize(tensor)
    
    return transform

