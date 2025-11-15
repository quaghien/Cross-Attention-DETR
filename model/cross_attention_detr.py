"""Cross-Attention DETR model for reference-based detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .backbone import build_backbone
from .transformer import (
    PositionEmbeddingSine,
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer
)


class CrossAttentionDETR(nn.Module):
    """
    Cross-Attention DETR for reference-based object detection.
    
    Architecture:
    1. Backbone extracts features from template and search images
    2. Template features are encoded and used as memory
    3. Search features are flattened and positionally encoded
    4. Decoder uses learnable queries that attend to search features (self-attn)
       and template features (cross-attn)
    5. Detection heads predict class and bbox for each query
    
    For single-object detection, we use num_queries=1 (one object per image).
    """
    
    def __init__(
        self,
        backbone_name: str = 'resnet18',
        num_queries: int = 1,
        hidden_dim: int = 256,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pretrained_backbone: bool = True
    ):
        """
        Args:
            backbone_name: 'resnet18' or 'efficientnet_b0'
            num_queries: Number of object queries (1 for single object)
            hidden_dim: Hidden dimension for transformer
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            num_heads: Number of attention heads
            dim_feedforward: FFN dimension
            dropout: Dropout rate
            pretrained_backbone: Use pretrained backbone
        """
        super().__init__()
        
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        # Backbone
        self.backbone = build_backbone(
            name=backbone_name,
            pretrained=pretrained_backbone,
            out_channels=hidden_dim
        )
        
        # Positional encoding
        self.pos_encoder = PositionEmbeddingSine(num_pos_feats=hidden_dim // 2)
        
        # Input projection (if needed)
        self.input_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        
        # Transformer encoder (optional, for search features)
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Transformer decoder (queries attend to template via cross-attention)
        decoder_layer = TransformerDecoderLayer(
            d_model=hidden_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Detection heads
        self.class_head = nn.Linear(hidden_dim, 1)  # Binary: object present or not
        self.bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)  # 4 bbox coords
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, template: torch.Tensor, search: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            template: (B, N, 3, H, W) - Multiple template/reference images (N=1-3)
            search: (B, 3, H, W) - Search images
            
        Returns:
            pred_logits: (B, num_queries, 1) - Classification logits
            pred_boxes: (B, num_queries, 4) - Predicted boxes in [cx, cy, w, h] format (normalized 0-1)
        """
        B, N = template.shape[0], template.shape[1]
        
        # Extract features from multiple templates (NO averaging!)
        # Reshape: (B, N, 3, H, W) -> (B*N, 3, H, W)
        template_flat = template.reshape(B * N, *template.shape[2:])
        template_feats = self.backbone(template_flat)  # (B*N, hidden_dim, H', W')
        
        # Reshape back: (B*N, C, H', W') -> (B, N, C, H', W')
        C, H_t, W_t = template_feats.shape[1:]
        template_feats = template_feats.reshape(B, N, C, H_t, W_t)
        
        # Extract search features
        search_feat = self.backbone(search)  # (B, hidden_dim, H', W')
        
        # Project search features
        search_feat = self.input_proj(search_feat)
        
        # Positional encoding for search
        search_pos = self.pos_encoder(search_feat)  # (B, hidden_dim, H', W')
        
        # Flatten search spatial dimensions: (B, C, H, W) -> (B, H*W, C)
        B, C, H_s, W_s = search_feat.shape
        search_flat = search_feat.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        search_pos_flat = search_pos.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        
        # Add positional encoding to search features
        search_with_pos = search_flat + search_pos_flat
        
        # Encode search features (self-attention)
        search_encoded = self.encoder(search_with_pos)  # (B, H*W, C)
        
        # Process ALL templates and concatenate them as memory
        # Each template becomes part of the memory that decoder attends to
        template_encoded_list = []
        for i in range(N):
            # Project and encode each template
            template_i = self.input_proj(template_feats[:, i])  # (B, C, H', W')
            template_pos_i = self.pos_encoder(template_i)  # (B, C, H', W')
            
            # Flatten: (B, C, H, W) -> (B, H*W, C)
            template_flat_i = template_i.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
            template_pos_flat_i = template_pos_i.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
            
            # Add positional encoding
            template_with_pos_i = template_flat_i + template_pos_flat_i
            
            # Encode (self-attention)
            template_encoded_i = self.encoder(template_with_pos_i)  # (B, H*W, C)
            template_encoded_list.append(template_encoded_i)
        
        # Concatenate all templates: (B, N*H*W, C)
        # Decoder will cross-attend to ALL template features
        template_encoded = torch.cat(template_encoded_list, dim=1)  # (B, N*H*W, C)
        
        # Concatenate template and search features as memory
        # Decoder will cross-attend to BOTH template and search
        memory = torch.cat([template_encoded, search_encoded], dim=1)  # (B, N*H*W + H*W, C)
        
        # Prepare queries
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # (B, num_queries, C)
        
        # Initialize decoder input with query embeddings
        tgt = torch.zeros_like(query_embed)  # (B, num_queries, C)
        
        # Decode with cross-attention to both template and search
        hs = self.decoder(tgt + query_embed, memory)  # (B, num_queries, C)
        
        # Prediction heads
        pred_logits = self.class_head(hs)  # (B, num_queries, 1)
        pred_boxes = self.bbox_head(hs).sigmoid()  # (B, num_queries, 4), normalized to [0, 1]
        
        return pred_logits, pred_boxes
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class MLP(nn.Module):
    """Simple multi-layer perceptron (FFN)."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_ca_detr(
    backbone_name: str = 'resnet18',
    num_queries: int = 1,
    hidden_dim: int = 256,
    num_encoder_layers: int = 3,
    num_decoder_layers: int = 3,
    num_heads: int = 8,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    pretrained_backbone: bool = True
) -> CrossAttentionDETR:
    """
    Build Cross-Attention DETR model.
    
    Recommended configs:
    - Lightweight: backbone='efficientnet_b0', hidden_dim=256, layers=2/2, heads=8
    - Balanced: backbone='resnet18', hidden_dim=256, layers=3/3, heads=8
    - High accuracy: backbone='resnet18', hidden_dim=256, layers=4/4, heads=8
    """
    return CrossAttentionDETR(
        backbone_name=backbone_name,
        num_queries=num_queries,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        pretrained_backbone=pretrained_backbone
    )


if __name__ == "__main__":
    # Test model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_ca_detr(
        backbone_name='resnet18',
        num_queries=1,
        hidden_dim=256,
        num_encoder_layers=3,
        num_decoder_layers=3,
        num_heads=8,
        pretrained_backbone=False
    ).to(device)
    
    # Test forward pass
    template = torch.randn(2, 3, 640, 640).to(device)
    search = torch.randn(2, 3, 640, 640).to(device)
    
    with torch.no_grad():
        pred_logits, pred_boxes = model(template, search)
    
    print(f"Model parameters: {model.get_num_params()/1e6:.2f}M")
    print(f"Template shape: {template.shape}")
    print(f"Search shape: {search.shape}")
    print(f"Pred logits shape: {pred_logits.shape}")
    print(f"Pred boxes shape: {pred_boxes.shape}")

