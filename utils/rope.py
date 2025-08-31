"""
Rotary Position Embedding (RoPE) implementation for Transformer models.

RoPE encodes absolute positional information with rotation matrices and naturally
incorporates relative position dependency in self-attention formulations.
Based on the paper: https://arxiv.org/abs/2104.09864

This implementation uses real-valued operations for full TorchInductor compatibility.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


def precompute_freqs(
    dim: int,
    max_position_embeddings: int = 2048,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
    scaling_factor: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute the cos and sin frequencies for rotary embeddings.
    
    Args:
        dim: Dimension of the model's head (hidden_size // num_heads)
        max_position_embeddings: Maximum sequence length to precompute
        theta: Base for the geometric progression of frequencies (typically 10000)
        device: Device to place the tensors on
        scaling_factor: Scaling factor for position indices (1.0 for no scaling)
        
    Returns:
        Tuple of (cos, sin) tensors of shape (max_position_embeddings, dim // 2)
    """
    # Compute the inverse frequencies
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    
    # Create position indices scaled by the scaling factor
    t = torch.arange(max_position_embeddings, device=device, dtype=torch.float32)
    t = t / scaling_factor
    
    # Compute outer product to get frequencies for each position
    freqs = torch.outer(t, freqs)
    
    # Return cos and sin separately for real-valued operations
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings using real-valued operations.
    
    This function applies 2D rotations to pairs of elements in the head dimension,
    which is mathematically equivalent to complex number rotation but uses only
    real arithmetic for full TorchInductor compatibility.
    
    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, num_heads, seq_len, head_dim)  
        freqs_cos: Cosine frequencies from precompute_freqs
        freqs_sin: Sine frequencies from precompute_freqs
        position_ids: Optional custom position IDs
        
    Returns:
        Tuple of rotated (q, k) tensors with same shapes as input
    """
    # q and k might have different num_heads (due to GQA)
    # q shape: (batch, num_q_heads, seq_len, head_dim)
    # k shape: (batch, num_kv_heads, seq_len, head_dim)
    batch_size, num_q_heads, seq_len, head_dim = q.shape
    _, num_kv_heads, _, _ = k.shape
    
    # Get frequencies for the current sequence length
    if position_ids is not None:
        freqs_cos = freqs_cos[position_ids]
        freqs_sin = freqs_sin[position_ids]
    else:
        freqs_cos = freqs_cos[:seq_len]
        freqs_sin = freqs_sin[:seq_len]
    
    # Reshape frequencies for broadcasting
    # From (seq_len, dim//2) to (1, 1, seq_len, dim//2)
    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(0)
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(0)
    
    # Cast frequencies to match input dtype for proper computation
    freqs_cos = freqs_cos.to(q.dtype)
    freqs_sin = freqs_sin.to(q.dtype)
    
    # Reshape q and k to separate pairs
    # From (..., head_dim) to (..., head_dim//2, 2)
    q_reshape = q.reshape(batch_size, num_q_heads, seq_len, head_dim // 2, 2)
    k_reshape = k.reshape(batch_size, num_kv_heads, seq_len, head_dim // 2, 2)
    
    # Extract pairs
    q_r = q_reshape[..., 0]  # Real parts (even indices)
    q_i = q_reshape[..., 1]  # Imaginary parts (odd indices)
    k_r = k_reshape[..., 0]
    k_i = k_reshape[..., 1]
    
    # Apply rotation using real arithmetic
    # Rotation formula: (r + i*i) * (cos + i*sin) = (r*cos - i*sin) + i*(r*sin + i*cos)
    q_rot_r = q_r * freqs_cos - q_i * freqs_sin
    q_rot_i = q_r * freqs_sin + q_i * freqs_cos
    k_rot_r = k_r * freqs_cos - k_i * freqs_sin
    k_rot_i = k_r * freqs_sin + k_i * freqs_cos
    
    # Stack and reshape back
    q_rot = torch.stack([q_rot_r, q_rot_i], dim=-1)
    k_rot = torch.stack([k_rot_r, k_rot_i], dim=-1)
    
    # Flatten last two dimensions to get back original shape
    q_rot = q_rot.reshape(batch_size, num_q_heads, seq_len, head_dim)
    k_rot = k_rot.reshape(batch_size, num_kv_heads, seq_len, head_dim)
    
    return q_rot, k_rot


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding module with TorchInductor-compatible real-valued operations.
    
    This module precomputes the frequencies at initialization and applies them during forward pass.
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        theta: float = 10000.0,
        scaling_factor: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the RotaryEmbedding module.
        
        Args:
            dim: Dimension of the model's head (hidden_size // num_heads)
            max_position_embeddings: Maximum sequence length to support
            theta: Base for the geometric progression of frequencies
            scaling_factor: Scaling factor for position indices
            device: Device to place the precomputed tensors on
        """
        super().__init__()
        
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta
        self.scaling_factor = scaling_factor
        
        # Precompute cos and sin frequencies for real-valued operations
        freqs_cos, freqs_sin = precompute_freqs(
            dim=dim,
            max_position_embeddings=max_position_embeddings,
            theta=theta,
            device=device,
            scaling_factor=scaling_factor
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        position_offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors.
        
        Args:
            q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
            k: Key tensor of shape (batch, num_heads, seq_len, head_dim)
            position_ids: Optional position IDs for custom positioning
            position_offset: Offset to add to sequential positions (for KV caching)
            
        Returns:
            Tuple of rotated (q, k) tensors
        """
        # If position_ids not provided, create sequential positions with offset
        if position_ids is None and position_offset > 0:
            seq_len = q.shape[2]
            position_ids = torch.arange(
                position_offset, 
                position_offset + seq_len, 
                device=q.device, 
                dtype=torch.long
            )
        
        return apply_rotary_pos_emb(q, k, self.freqs_cos, self.freqs_sin, position_ids)
