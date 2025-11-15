"""Transformer components for CA-DETR."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class PositionEmbeddingSine(nn.Module):
    """
    2D positional encoding using sine/cosine functions.
    Similar to the original Transformer paper but extended to 2D.
    """
    
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000, normalize: bool = True, scale: Optional[float] = None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) - Feature maps
            
        Returns:
            pos: (B, num_pos_feats*2, H, W) - Positional encoding
        """
        B, C, H, W = x.shape
        
        # Create coordinate grids
        y_embed = torch.arange(H, dtype=torch.float32, device=x.device)
        x_embed = torch.arange(W, dtype=torch.float32, device=x.device)
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (H + eps) * self.scale
            x_embed = x_embed / (W + eps) * self.scale
        
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        
        pos = torch.cat((pos_y[:, None, :].repeat(1, W, 1), 
                        pos_x[None, :, :].repeat(H, 1, 1)), dim=-1).permute(2, 0, 1)
        
        pos = pos.unsqueeze(0).repeat(B, 1, 1, 1)
        return pos


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention."""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model) - Input features
            mask: Optional attention mask
            
        Returns:
            out: (B, N, d_model) - Output features
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        
        return out


class CrossAttention(nn.Module):
    """Cross-attention: Query from one source, Key/Value from another."""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key_value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, d_model) - Query features (e.g., search image)
            key_value: (B, N_kv, d_model) - Key/Value features (e.g., template)
            mask: Optional attention mask
            
        Returns:
            out: (B, N_q, d_model) - Output features
        """
        B, N_q, C = query.shape
        N_kv = key_value.shape[1]
        
        # Project Q, K, V
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Cross attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply to values
        out = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        out = self.out_proj(out)
        
        return out


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with self-attention and FFN."""
    
    def __init__(self, d_model: int, num_heads: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        src2 = self.self_attn(self.norm1(src), mask)
        src = src + self.dropout1(src2)
        
        # FFN
        src2 = self.linear2(self.dropout(F.relu(self.linear1(self.norm2(src)))))
        src = src + self.dropout2(src2)
        
        return src


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with self-attention, cross-attention, and FFN."""
    
    def __init__(self, d_model: int, num_heads: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = CrossAttention(d_model, num_heads, dropout)
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, 
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tgt: (B, N_q, d_model) - Target/query features
            memory: (B, N_m, d_model) - Memory/source features (from encoder or template)
            tgt_mask: Optional mask for self-attention
            memory_mask: Optional mask for cross-attention
            
        Returns:
            tgt: (B, N_q, d_model) - Output features
        """
        # Self-attention
        tgt2 = self.self_attn(self.norm1(tgt), tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        
        # Cross-attention with memory (template features)
        tgt2 = self.cross_attn(self.norm2(tgt), memory, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        
        # FFN
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(self.norm3(tgt)))))
        tgt = tgt + self.dropout3(tgt2)
        
        return tgt


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers."""
    
    def __init__(self, encoder_layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
    
    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, mask)
        return output


class TransformerDecoder(nn.Module):
    """Stack of transformer decoder layers."""
    
    def __init__(self, decoder_layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
    
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask)
        return output


if __name__ == "__main__":
    # Test components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test positional encoding
    pos_enc = PositionEmbeddingSine(num_pos_feats=128).to(device)
    x = torch.randn(2, 256, 20, 20).to(device)
    pos = pos_enc(x)
    print(f"Positional encoding: input {x.shape} -> output {pos.shape}")
    
    # Test cross-attention
    cross_attn = CrossAttention(d_model=256, num_heads=8).to(device)
    query = torch.randn(2, 100, 256).to(device)  # Search features
    kv = torch.randn(2, 50, 256).to(device)      # Template features
    out = cross_attn(query, kv)
    print(f"Cross-attention: query {query.shape}, kv {kv.shape} -> output {out.shape}")
    
    # Test decoder layer
    decoder_layer = TransformerDecoderLayer(d_model=256, num_heads=8).to(device)
    tgt = torch.randn(2, 100, 256).to(device)
    memory = torch.randn(2, 50, 256).to(device)
    out = decoder_layer(tgt, memory)
    print(f"Decoder layer: tgt {tgt.shape}, memory {memory.shape} -> output {out.shape}")

