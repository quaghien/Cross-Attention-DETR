"""Lightweight backbones for CA-DETR."""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict


class BackboneBase(nn.Module):
    """Base class for backbone networks."""
    
    def __init__(self, backbone: nn.Module, num_channels: int, return_layers: Dict[str, str]):
        super().__init__()
        self.body = backbone
        self.num_channels = num_channels
        self.return_layers = return_layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input image."""
        return self.body(x)


class ResNet18Backbone(nn.Module):
    """ResNet-18 backbone for feature extraction.
    
    Output: (B, 512, H/32, W/32) where H, W are input dimensions.
    For 640x640 input -> (B, 512, 20, 20)
    """
    
    def __init__(self, pretrained: bool = True, out_channels: int = 256):
        super().__init__()
        # Load pretrained ResNet-18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remove avgpool and fc layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels
        
        # Projection to reduce channels
        self.projection = nn.Conv2d(512, out_channels, kernel_size=1)
        self.num_channels = out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) - Input images
            
        Returns:
            features: (B, out_channels, H/32, W/32) - Feature maps
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.projection(x)
        return x


class EfficientNetB0Backbone(nn.Module):
    """EfficientNet-B0 backbone for feature extraction.
    
    Lightweight and efficient, ~5M parameters.
    Output: (B, 256, H/32, W/32)
    """
    
    def __init__(self, pretrained: bool = True, out_channels: int = 256):
        super().__init__()
        # Load pretrained EfficientNet-B0
        efficientnet = models.efficientnet_b0(pretrained=pretrained)
        
        # Extract feature layers
        self.features = efficientnet.features
        
        # Get output channels from last layer
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out = self.features(dummy)
            in_channels = out.shape[1]
        
        # Projection to target channels
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.num_channels = out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) - Input images
            
        Returns:
            features: (B, out_channels, H/32, W/32) - Feature maps
        """
        x = self.features(x)
        x = self.projection(x)
        return x


def build_backbone(name: str = 'resnet18', pretrained: bool = True, out_channels: int = 256) -> nn.Module:
    """
    Build backbone network.
    
    Args:
        name: Backbone name ('resnet18' or 'efficientnet_b0')
        pretrained: Use pretrained weights
        out_channels: Output feature channels
        
    Returns:
        backbone: Backbone network
    """
    if name == 'resnet18':
        return ResNet18Backbone(pretrained=pretrained, out_channels=out_channels)
    elif name == 'efficientnet_b0':
        return EfficientNetB0Backbone(pretrained=pretrained, out_channels=out_channels)
    else:
        raise ValueError(f"Unknown backbone: {name}. Choose 'resnet18' or 'efficientnet_b0'")


if __name__ == "__main__":
    # Test backbones
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for backbone_name in ['resnet18', 'efficientnet_b0']:
        print(f"\nTesting {backbone_name}...")
        backbone = build_backbone(backbone_name, pretrained=False, out_channels=256).to(device)
        
        # Test with 640x640 input
        x = torch.randn(2, 3, 640, 640).to(device)
        with torch.no_grad():
            out = backbone(x)
        
        params = sum(p.numel() for p in backbone.parameters())
        print(f"  Input: {x.shape}")
        print(f"  Output: {out.shape}")
        print(f"  Parameters: {params/1e6:.2f}M")

