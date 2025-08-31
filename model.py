#!/usr/bin/env python3
"""
Transformer Language Model Implementation

This module defines a standard transformer architecture for language modeling,
using native PyTorch modules for efficiency and simplicity.
"""

import json
import math
import os
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# Import our custom modules
from utils.rope import RotaryEmbedding
from utils.activations import SwiGLU

# Set environment variable for optimal memory allocation on GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Enable TF32 for better performance with high precision mode on Ampere+
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')  # More aggressive TF32 for ~10-15% speedup

class ScaledResidual(nn.Module):
    """
    Learnable or fixed residual scaling to prevent gradient explosion in deep networks.
    Scales residual connections by a factor that depends on layer depth.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        layer_idx: int,
        learnable: bool = False
    ):
        super().__init__()
        
        # Calculate optimal scaling factor based on layer depth
        # Scale by 1/sqrt(2n) where n = layer index + 1
        scale_value = math.sqrt(1.0 / (2 * (layer_idx + 1)))
        
        if learnable:
            # Learnable scaling parameter (initialized to optimal value)
            self.scale = nn.Parameter(torch.ones(1) * scale_value)
        else:
            # Fixed scaling based on layer depth
            self.register_buffer('scale', torch.tensor(scale_value))
    
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Apply scaled residual connection."""
        return residual + self.scale * x

class FastMultiHeadAttention(nn.Module):
    """
    Optimized Multi-Head Attention with optional Grouped Query Attention (GQA) support.
    Uses RoPE for position encoding and optional Flash Attention for maximum performance.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        max_position_embeddings: int = 512,
        rope_theta: float = 10000.0,
        rope_scaling: float = 1.0,
        use_gqa: bool = True,
        use_flash_attention: bool = True
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_gqa = use_gqa
        self.use_flash_attention = use_flash_attention
        
        # Handle GQA configuration
        if use_gqa and num_kv_heads is not None:
            self.num_kv_heads = num_kv_heads
        else:
            # If GQA is disabled or num_kv_heads not specified, use standard MHA
            self.num_kv_heads = num_heads
        
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Ensure num_heads is divisible by num_kv_heads for even grouping
        assert self.num_heads % self.num_kv_heads == 0, \
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        self.num_groups = self.num_heads // self.num_kv_heads
        
        # Fully fused QKV projection: single GEMM producing Q, K, and V
        # Output size is (num_heads + 2 * num_kv_heads) * head_dim
        # This reduces kernel launches and improves memory bandwidth utilization
        self.qkv_proj = nn.Linear(
            hidden_size, 
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim, 
            bias=False  # No bias for better kernel fusion
        )
        
        # Mark attention projection for specialized initialization
        self.qkv_proj._is_attention_qkv = True
        
        # Output projection
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)  # No bias for better kernel fusion
        self.o_proj._is_attention_output = True
        
        self.dropout = dropout
        
        # Initialize RoPE (TorchInductor-compatible real-valued operations)
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            theta=rope_theta,
            scaling_factor=rope_scaling,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Single fused QKV projection - reduces 2 GEMMs to 1
        # Output: (batch, seq_len, (num_heads + 2 * num_kv_heads) * head_dim)
        qkv = self.qkv_proj(hidden_states)
        
        # Split QKV into Q and KV components
        # Q size: num_heads * head_dim
        # KV size: 2 * num_kv_heads * head_dim
        q_size = self.num_heads * self.head_dim
        kv_size = 2 * self.num_kv_heads * self.head_dim
        q, kv = qkv.split([q_size, kv_size], dim=-1)
        
        # Split KV into separate K and V tensors
        # Each will be (batch, seq_len, num_kv_heads * head_dim)
        k, v = kv.chunk(2, dim=-1)
        
        # Reshape Q to (batch, num_heads, seq_len, head_dim)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        
        # Reshape K, V to (batch, num_kv_heads, seq_len, head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        k = k.transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.transpose(1, 2)
        
        # Determine the position offset for RoPE
        past_seq_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_seq_len = past_k.shape[2]  # Get sequence length from past keys
        
        # Apply RoPE to Q and K with correct position offset
        # When using cache, positions should start from past_seq_len
        q, k = self.rotary_emb(q, k, position_offset=past_seq_len)
        
        # Concatenate with past key values if provided
        if past_key_value is not None:
            k = torch.cat([past_k, k], dim=2)  # Concatenate along sequence dimension
            v = torch.cat([past_v, v], dim=2)
        
        # Cache the current key and value states if requested
        present_key_value = None
        if use_cache:
            # Store K and V before expansion for memory efficiency
            present_key_value = (k, v)
        
        # Expand K and V to match Q heads for GQA
        if self.num_kv_heads != self.num_heads:
            # Use expand instead of repeat_interleave for memory efficiency
            # This creates a view without copying data - SDPA can handle strided tensors
            # Shape: (batch, num_kv_heads, seq_len, head_dim) -> (batch, num_heads, seq_len, head_dim)
            batch_size, num_kv_heads, kv_seq_len, head_dim = k.shape
            
            # Reshape to add a new dimension for groups, then expand
            k = k.reshape(batch_size, num_kv_heads, 1, kv_seq_len, head_dim)
            k = k.expand(batch_size, num_kv_heads, self.num_groups, kv_seq_len, head_dim)
            k = k.reshape(batch_size, self.num_heads, kv_seq_len, head_dim)
            
            v = v.reshape(batch_size, num_kv_heads, 1, kv_seq_len, head_dim)
            v = v.expand(batch_size, num_kv_heads, self.num_groups, kv_seq_len, head_dim)
            v = v.reshape(batch_size, self.num_heads, kv_seq_len, head_dim)
        
        # Use Flash Attention backend for optimal performance when requested and available
        if self.use_flash_attention and torch.cuda.is_available():
            try:
                # Try to use Flash Attention backend
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    attn_output = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=attention_mask,
                        dropout_p=self.dropout if self.training else 0.0,
                        is_causal=is_causal and attention_mask is None,
                        scale=self.scale
                    )
            except RuntimeError:
                # Fallback to efficient attention if Flash Attention fails
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    attn_output = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=attention_mask,
                        dropout_p=self.dropout if self.training else 0.0,
                        is_causal=is_causal and attention_mask is None,
                        scale=self.scale
                    )
        else:
            # Standard SDPA - let PyTorch choose the best backend
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal and attention_mask is None,
                scale=self.scale
            )
        
        # Reshape back to (batch, seq_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        # Return output with optional cache
        if use_cache:
            return output, present_key_value
        return output

class FastTransformerDecoderLayer(nn.Module):
    """
    Optimized Transformer Decoder Layer using FastMultiHeadAttention with RoPE.
    Uses SwiGLU activation and RMSNorm with pre-norm architecture for better training stability.
    Includes optional residual scaling and gradient checkpointing support.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        intermediate_size: int = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        max_position_embeddings: int = 512,
        rope_theta: float = 10000.0,
        rope_scaling: float = 1.0,
        layer_idx: int = 0,
        num_layers: int = 1,
        use_scaled_residuals: bool = True,
        use_gradient_checkpointing: bool = False,
        use_gqa: bool = True,
        use_flash_attention: bool = True
    ):
        super().__init__()
        
        if intermediate_size is None:
            intermediate_size = hidden_size * 4
        
        # Self-attention components with optional GQA and RoPE
        self.self_attn = FastMultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=attention_dropout,
            bias=False,  # Always False for optimized kernel fusion
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            use_gqa=use_gqa,
            use_flash_attention=use_flash_attention
        )
        
        # Feed-forward network - always use SwiGLU for better gradient flow
        self.ffn = SwiGLU(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=False,  # No bias for better kernel fusion and memory savings
            dropout=dropout
        )
        
        # Always use RMSNorm for better efficiency
        self.norm1 = nn.RMSNorm(hidden_size, eps=layer_norm_eps)
        self.norm2 = nn.RMSNorm(hidden_size, eps=layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Residual scaling for deep networks
        self.use_scaled_residuals = use_scaled_residuals
        if use_scaled_residuals:
            self.residual_scale_attn = ScaledResidual(
                hidden_size, num_layers, layer_idx, learnable=True
            )
            self.residual_scale_ffn = ScaledResidual(
                hidden_size, num_layers, layer_idx, learnable=True
            )
        
        # Gradient checkpointing support
        self.use_gradient_checkpointing = use_gradient_checkpointing
    
    def _attention_block(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]]:
        """Attention sub-block for gradient checkpointing."""
        # Pre-norm architecture: LayerNorm before attention
        hidden_states = self.norm1(hidden_states)
        
        # Self-attention with optional KV cache
        attn_outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            is_causal=is_causal,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        
        # Handle both cached and non-cached outputs
        if use_cache:
            hidden_states, present_key_value = attn_outputs
        else:
            hidden_states = attn_outputs
            present_key_value = None
            
        hidden_states = self.dropout(hidden_states)
        
        if use_cache:
            return hidden_states, present_key_value
        return hidden_states
    
    def _ffn_block(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Feed-forward sub-block for gradient checkpointing."""
        # Pre-norm
        hidden_states = self.norm2(hidden_states)
        
        # Always use SwiGLU activation
        hidden_states = self.ffn(hidden_states)
        
        return hidden_states
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]]:
        # Note: Gradient checkpointing is not compatible with KV caching
        # When use_cache is True, we disable gradient checkpointing for this layer
        if self.use_gradient_checkpointing and self.training and not use_cache:
            # Checkpoint attention block (without cache)
            residual = hidden_states
            attn_output = torch.utils.checkpoint.checkpoint(
                self._attention_block,
                hidden_states,
                attention_mask,
                is_causal,
                None,  # No past_key_value in gradient checkpointing
                False,  # No use_cache in gradient checkpointing
                use_reentrant=False
            )
            
            # Apply residual with optional scaling
            if self.use_scaled_residuals:
                hidden_states = self.residual_scale_attn(attn_output, residual)
            else:
                hidden_states = residual + attn_output
            
            # Checkpoint FFN block
            residual = hidden_states
            ffn_output = torch.utils.checkpoint.checkpoint(
                self._ffn_block,
                hidden_states,
                use_reentrant=False
            )
            
            # Apply residual with optional scaling
            if self.use_scaled_residuals:
                hidden_states = self.residual_scale_ffn(ffn_output, residual)
            else:
                hidden_states = residual + ffn_output
                
            # No cache to return when using gradient checkpointing
            return hidden_states
        else:
            # Normal forward pass (with optional KV caching)
            # Attention block
            residual = hidden_states
            attn_outputs = self._attention_block(
                hidden_states, 
                attention_mask, 
                is_causal,
                past_key_value,
                use_cache
            )
            
            # Handle both cached and non-cached outputs
            if use_cache:
                attn_output, present_key_value = attn_outputs
            else:
                attn_output = attn_outputs
                present_key_value = None
            
            # Apply residual with optional scaling
            if self.use_scaled_residuals:
                hidden_states = self.residual_scale_attn(attn_output, residual)
            else:
                hidden_states = residual + attn_output
            
            # FFN block
            residual = hidden_states
            ffn_output = self._ffn_block(hidden_states)
            
            # Apply residual with optional scaling
            if self.use_scaled_residuals:
                hidden_states = self.residual_scale_ffn(ffn_output, residual)
            else:
                hidden_states = residual + ffn_output
        
            # Return with optional cache
            if use_cache:
                return hidden_states, present_key_value
            return hidden_states

class FastTransformerDecoder(nn.Module):
    """
    Stack of optimized Transformer Decoder layers with RoPE, SwiGLU, and RMSNorm.
    Eliminates unnecessary operations from nn.TransformerDecoder.
    """
    
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        max_position_embeddings: int = 512,
        rope_theta: float = 10000.0,
        rope_scaling: float = 1.0,
        use_scaled_residuals: bool = True,
        use_gradient_checkpointing: bool = False,
        use_gqa: bool = True,
        use_flash_attention: bool = True
    ):
        super().__init__()
        
        if intermediate_size is None:
            intermediate_size = hidden_size * 4
        
        # Stack of decoder layers with GQA, RoPE, SwiGLU, and RMSNorm support
        self.layers = nn.ModuleList([
            FastTransformerDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                attention_dropout=attention_dropout,
                layer_norm_eps=layer_norm_eps,
                bias=False,  # Always False for optimized kernel fusion
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                layer_idx=i,
                num_layers=num_layers,
                use_scaled_residuals=use_scaled_residuals,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gqa=use_gqa,
                use_flash_attention=use_flash_attention
            )
            for i in range(num_layers)
        ])
        
        # Final layer norm - always use RMSNorm for efficiency
        self.norm = nn.RMSNorm(hidden_size, eps=layer_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]]:
        # Initialize past_key_values if not provided
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        
        # Will store present key values if caching is enabled
        present_key_values = [] if use_cache else None
        
        # Process through each layer
        for idx, layer in enumerate(self.layers):
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                is_causal=is_causal,
                past_key_value=past_key_values[idx],
                use_cache=use_cache
            )
            
            # Handle both cached and non-cached outputs
            if use_cache:
                hidden_states, present_key_value = layer_outputs
                present_key_values.append(present_key_value)
            else:
                hidden_states = layer_outputs
        
        # Apply final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Return with optional caches
        if use_cache:
            return hidden_states, present_key_values
        return hidden_states

class TransformerLM(nn.Module):
    """Transformer Language Model with RoPE, SwiGLU, and RMSNorm."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        num_kv_heads: Optional[int] = None,
        intermediate_size: Optional[int] = None,  # Allow explicit intermediate size
        max_position_embeddings: int = 512,
        rope_theta: float = 10000.0,
        rope_scaling: float = 1.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        pad_token_id: int = 2,  # Same as EOS token (CodeLlama convention)
        eos_token_id: int = 2,
        use_scaled_residuals: bool = True,
        use_gradient_checkpointing: bool = False,
        use_gqa: bool = True,
        use_flash_attention: bool = True,
        tie_embeddings: bool = True
    ):
        super().__init__()
        
        # Calculate intermediate size if not provided
        if intermediate_size is None:
            intermediate_size = hidden_size * 3
        
        # Default num_kv_heads to num_heads if not specified (backward compatibility)
        if num_kv_heads is None:
            num_kv_heads = num_heads
        
        self.config = {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'num_kv_heads': num_kv_heads,
            'intermediate_size': intermediate_size,
            'max_position_embeddings': max_position_embeddings,
            'rope_theta': rope_theta,
            'rope_scaling': rope_scaling,
            'dropout': dropout,
            'attention_dropout': attention_dropout,
            'layer_norm_eps': layer_norm_eps,
            'initializer_range': initializer_range,
            'use_cache': use_cache,
            'pad_token_id': pad_token_id,
            'eos_token_id': eos_token_id,
            'use_scaled_residuals': use_scaled_residuals,
            'use_gradient_checkpointing': use_gradient_checkpointing,
            'use_gqa': use_gqa,
            'use_flash_attention': use_flash_attention,
            'tie_embeddings': tie_embeddings
        }
        
        # Token embeddings only (RoPE handles position encoding)
        self.embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        
        # Embedding dropout
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Use optimized FastTransformerDecoder with configurable features
        self.transformer = FastTransformerDecoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            intermediate_size=intermediate_size,
            dropout=dropout,
            attention_dropout=attention_dropout,
            layer_norm_eps=layer_norm_eps,
            bias=True,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            use_scaled_residuals=use_scaled_residuals,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gqa=use_gqa,
            use_flash_attention=use_flash_attention
        )
        
        # Language modeling head (optionally tied with input embeddings)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Optionally tie weights between input and output embeddings
        if tie_embeddings:
            self.lm_head.weight = self.embeddings.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Always move model to GPU if available
        if torch.cuda.is_available():
            self.cuda()
    
    def _init_weights(self, module):
        """Initialize model weights with enhanced, module-specific strategies."""
        
        # Calculate depth-dependent scaling for output projections
        num_layers = self.config['num_layers']
        depth_scale = 1.0 / math.sqrt(2 * num_layers) if num_layers > 1 else 1.0
        
        if isinstance(module, nn.Linear):
            # Check if this is part of attention mechanism
            if hasattr(module, '_is_attention_qkv'):
                # Query, Key, Value projections - use scaled initialization
                fan_in = module.weight.size(1)
                fan_out = module.weight.size(0)
                std = math.sqrt(2.0 / (fan_in + fan_out))
                module.weight.data.normal_(mean=0.0, std=std)
                
            elif hasattr(module, '_is_attention_output'):
                # Attention output projection - scale by depth
                std = self.config['initializer_range'] * depth_scale
                module.weight.data.normal_(mean=0.0, std=std)
                
            elif isinstance(module.weight, nn.Parameter) and module.out_features > module.in_features * 2:
                # Likely FFN up/gate projection - use Xavier uniform
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                
            elif isinstance(module.weight, nn.Parameter) and module.in_features > module.out_features * 2:
                # Likely FFN down projection - scale by depth
                std = self.config['initializer_range'] * depth_scale
                module.weight.data.normal_(mean=0.0, std=std)
                
            else:
                # Default initialization for other linear layers
                module.weight.data.normal_(mean=0.0, std=self.config['initializer_range'])
            
            # Initialize bias to zero if present
            if module.bias is not None:
                module.bias.data.zero_()
                
        elif isinstance(module, nn.Embedding):
            # Token embeddings with normal distribution
            module.weight.data.normal_(mean=0.0, std=self.config['initializer_range'])
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
                
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            # Handle both LayerNorm and RMSNorm
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
        elif isinstance(module, SwiGLU):
            # Special initialization for SwiGLU components
            # Fused gate+up projection: Xavier uniform
            if hasattr(module, 'w_fused'):
                nn.init.xavier_uniform_(module.w_fused.weight, gain=1.0)
                if module.w_fused.bias is not None:
                    module.w_fused.bias.data.zero_()
                    
            # Down projection: scaled by depth
            if hasattr(module, 'w_down'):
                std = self.config['initializer_range'] * depth_scale
                module.w_down.weight.data.normal_(mean=0.0, std=std)
                if module.w_down.bias is not None:
                    module.w_down.bias.data.zero_()
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory-efficient training."""
        # Set flag for all decoder layers
        for layer in self.transformer.layers:
            layer.use_gradient_checkpointing = True
        
        # Update config
        self.config['use_gradient_checkpointing'] = True
        
        # Log memory savings estimate
        if self.training:
            seq_len = self.config['max_position_embeddings']
            batch_size = 8  # Example batch size
            memory_saved_mb = self._estimate_memory_savings(batch_size, seq_len)
            print(f"✓ Gradient checkpointing enabled")
            print(f"  Estimated memory savings: ~{memory_saved_mb:.1f} MB per batch")
            print(f"  Trade-off: ~30% slower training for 40-50% memory reduction")
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing for faster training."""
        for layer in self.transformer.layers:
            layer.use_gradient_checkpointing = False
        
        # Update config
        self.config['use_gradient_checkpointing'] = False
        print("✓ Gradient checkpointing disabled")
    
    def _estimate_memory_savings(self, batch_size: int, seq_len: int) -> float:
        """Estimate memory savings from gradient checkpointing."""
        hidden_size = self.config['hidden_size']
        num_layers = self.config['num_layers']
        intermediate_size = self.config.get('intermediate_size', hidden_size * 4)
        
        # Activations per layer (attention + FFN intermediates)
        # Attention: Q, K, V, attention scores, attention output
        # FFN: intermediate activations
        acts_per_layer = (
            batch_size * seq_len * hidden_size * 4 *  # Attention intermediates
            (5 + intermediate_size / hidden_size)      # FFN intermediates
        )
        
        # With checkpointing: only store sqrt(n) activations
        saved = acts_per_layer * num_layers * (1 - 1/math.sqrt(num_layers))
        return saved / (1024 * 1024)  # Convert to MB
    
    def configure_gradient_checkpointing(self, policy: str = "adaptive", memory_threshold_gb: float = 10.0):
        """
        Configure gradient checkpointing based on policy.
        
        Policies:
        - 'none': No checkpointing
        - 'all': Checkpoint all layers
        - 'adaptive': Enable based on model size and available memory
        - 'alternating': Checkpoint every other layer
        
        Args:
            policy: Checkpointing policy to use
            memory_threshold_gb: Memory threshold for adaptive policy
        """
        if policy == "none":
            self.disable_gradient_checkpointing()
        
        elif policy == "all":
            self.enable_gradient_checkpointing()
        
        elif policy == "adaptive":
            # Check available GPU memory
            if torch.cuda.is_available():
                free_memory_gb = torch.cuda.mem_get_info()[0] / (1024**3)
                model_memory_gb = sum(p.numel() * 4 for p in self.parameters()) / (1024**3)
                
                # Enable if model is large relative to available memory
                if model_memory_gb > free_memory_gb * 0.3 or model_memory_gb > memory_threshold_gb:
                    self.enable_gradient_checkpointing()
                    print(f"Auto-enabled gradient checkpointing")
                    print(f"  Model size: {model_memory_gb:.1f}GB")
                    print(f"  Free memory: {free_memory_gb:.1f}GB")
                else:
                    self.disable_gradient_checkpointing()
                    print(f"Gradient checkpointing not needed")
                    print(f"  Model size: {model_memory_gb:.1f}GB")
                    print(f"  Free memory: {free_memory_gb:.1f}GB")
            else:
                # CPU mode - checkpointing less useful
                self.disable_gradient_checkpointing()
        
        elif policy == "alternating":
            # Checkpoint every other layer for balanced trade-off
            for i, layer in enumerate(self.transformer.layers):
                layer.use_gradient_checkpointing = (i % 2 == 0)
            
            self.config['use_gradient_checkpointing'] = 'alternating'
            print(f"✓ Alternating gradient checkpointing enabled")
            print(f"  Checkpointed layers: {len([l for i, l in enumerate(self.transformer.layers) if i % 2 == 0])}/{len(self.transformer.layers)}")
            print(f"  Trade-off: ~15% slower training for 20-25% memory reduction")
        
        else:
            raise ValueError(f"Unknown checkpointing policy: {policy}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        return_dict: bool = True
    ) -> Union[dict, tuple]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask for padding (batch_size, seq_len)
            labels: Target token IDs for language modeling loss
            past_key_values: Cached key-value states from previous forward passes
            use_cache: Whether to return cached key-value states
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary with 'logits', 'loss' (if labels provided), and 'past_key_values' (if use_cache)
        """
        # Get token embeddings (RoPE handles position encoding internally)
        hidden_states = self.embeddings(input_ids)
        hidden_states = self.embedding_dropout(hidden_states)
        
        # Pass through optimized transformer decoder with RoPE and SwiGLU
        # The FastTransformerDecoder handles causal masking and position encoding internally
        transformer_outputs = self.transformer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            is_causal=True,  # Always use causal masking for language modeling
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        
        # Handle both cached and non-cached outputs
        if use_cache:
            hidden_states, present_key_values = transformer_outputs
        else:
            hidden_states = transformer_outputs
            present_key_values = None
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Create mask for positions after the first EOS token
            eos_token_id = self.config.get('eos_token_id', 2)
            
            # Find positions of EOS tokens (before shifting)
            eos_positions = (labels == eos_token_id).float()
            
            # Use cumsum to identify positions after the first EOS
            # cumsum will be 0 before first EOS, 1 at first EOS, >1 after
            eos_cumsum = eos_positions.cumsum(dim=1)
            
            # For the loss calculation after shifting:
            # - We want to predict UP TO and INCLUDING the first EOS
            # - We DON'T want to predict anything AFTER the first EOS
            
            # Create base mask: True only before any EOS
            before_eos_mask = (eos_cumsum == 0)
            
            # After shifting for next-token prediction:
            # Position i in shift space predicts token at position i+1 in original space
            shift_mask = before_eos_mask[..., :-1].contiguous()
            
            # Find the SINGLE position that predicts the first EOS
            # This is where cumsum transitions from 0 to 1
            # Check if current position has cumsum==0 AND next position has cumsum==1
            eos_predictor = (eos_cumsum[..., :-1] == 0) & (eos_cumsum[..., 1:] == 1)
            
            # Combine: include positions before EOS AND the position predicting EOS
            shift_mask = shift_mask | eos_predictor
            
            # Replace labels after first EOS with -100 (ignore_index)
            # This ensures the model learns to predict EOS but not what comes after
            masked_labels = torch.where(
                shift_mask,
                shift_labels,
                torch.tensor(-100, device=shift_labels.device)
            )
            
            # Also mask out actual padding tokens (same as EOS in CodeLlama)
            pad_token_id = self.config.get('pad_token_id', 2)
            masked_labels = torch.where(
                shift_labels == pad_token_id,
                torch.tensor(-100, device=shift_labels.device),
                masked_labels
            )
            
            # Compute cross-entropy loss with proper masking
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config['vocab_size']),
                masked_labels.view(-1),
                ignore_index=-100
            )
        
        if return_dict:
            output = {
                'logits': logits,
                'loss': loss,
                'hidden_states': hidden_states
            }
            if use_cache:
                output['past_key_values'] = present_key_values
            return output
        else:
            if use_cache:
                return (logits, loss, present_key_values)
            else:
                return (logits, loss)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Generate text autoregressively with optional KV caching for massive speedup.
        
        Args:
            input_ids: Starting token IDs (batch_size, seq_len)
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID
            use_cache: Whether to use KV caching for speedup (recommended)
            
        Returns:
            Generated token IDs
        """
        self.eval()
        pad_token_id = pad_token_id or self.config['pad_token_id']
        eos_token_id = eos_token_id or self.config.get('eos_token_id', 2)
        
        # Initialize with input
        generated = input_ids
        batch_size = input_ids.shape[0]
        
        # Track which sequences are done
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        
        # Initialize KV cache (will be populated on first forward pass)
        past_key_values = None
        
        # Use bfloat16 autocast for generation (matches training precision)
        # This ensures Flash Attention and other optimized kernels work correctly
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            while generated.shape[1] < max_length:
                # Prepare model inputs
                if past_key_values is None:
                    # First iteration: process all input tokens
                    model_inputs = generated
                else:
                    # Subsequent iterations: only process the last generated token
                    # This is the key optimization - we only process 1 token instead of the full sequence
                    model_inputs = generated[:, -1:] 
                
                # Forward pass with KV caching
                outputs = self.forward(
                    model_inputs, 
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    return_dict=True
                )
                assert isinstance(outputs, dict)  # Type assertion for type checker
                
                # Extract logits for the last position
                next_token_logits = outputs['logits'][:, -1, :]
                
                # Update KV cache for next iteration
                if use_cache:
                    past_key_values = outputs.get('past_key_values')
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k is not None and top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep at least one token
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter back to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next tokens
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
                
                # Update tokens for unfinished sequences
                next_tokens = next_tokens * unfinished_sequences.unsqueeze(-1)
                # Use padding for finished sequences
                next_tokens = next_tokens + pad_token_id * (1 - unfinished_sequences).unsqueeze(-1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_tokens], dim=-1)
                
                # Update which sequences are finished
                unfinished_sequences = unfinished_sequences * (next_tokens.squeeze(-1) != eos_token_id).long()
                
                # Stop if all sequences are done
                if unfinished_sequences.sum() == 0:
                    break
        
        return generated
    
    def clear_cache(self) -> None:
        """
        Clear any cached key-value states to free memory.
        This is useful when switching between different generation tasks.
        """
        # Force garbage collection of any cached tensors
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_cache_memory_usage(self, batch_size: int = 1, seq_len: int = 512) -> dict:
        """
        Estimate memory usage of KV cache for given batch size and sequence length.
        
        Args:
            batch_size: Batch size for generation
            seq_len: Maximum sequence length expected
            
        Returns:
            Dictionary with memory usage estimates
        """
        num_layers = self.config['num_layers']
        num_kv_heads = self.config['num_kv_heads']
        head_dim = self.config['hidden_size'] // self.config['num_heads']
        
        # Each layer stores K and V tensors
        # Shape per tensor: (batch_size, num_kv_heads, seq_len, head_dim)
        elements_per_layer = 2 * batch_size * num_kv_heads * seq_len * head_dim
        total_elements = elements_per_layer * num_layers
        
        # Memory in bytes (assuming bfloat16/float16 = 2 bytes per element)
        bytes_per_element = 2
        total_bytes = total_elements * bytes_per_element
        
        return {
            'total_elements': total_elements,
            'memory_mb': total_bytes / (1024 * 1024),
            'memory_gb': total_bytes / (1024 * 1024 * 1024),
            'per_layer_mb': (elements_per_layer * bytes_per_element) / (1024 * 1024),
            'num_layers': num_layers
        }
    
    def initialize_cache(self, batch_size: int = 1) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Initialize an empty KV cache for generation.
        
        Args:
            batch_size: Batch size for generation
            
        Returns:
            List of empty cache tuples, one per layer
        """
        # For now, return None list - cache will be built incrementally
        # This is a placeholder for potential future optimizations
        return [None] * self.config['num_layers']
    
    def get_num_params(self) -> dict:
        """Get detailed breakdown of model parameters."""
        # Token embeddings (shared with output layer due to tying)
        embedding_params = self.embeddings.weight.numel()
        
        # No positional embeddings anymore (RoPE is parameter-free)
        positional_params = 0
        
        # Calculate parameters for a single transformer block
        hidden_size = self.config['hidden_size']
        intermediate_size = self.config['intermediate_size']
        num_heads = self.config['num_heads']
        num_kv_heads = self.config['num_kv_heads']
        head_dim = hidden_size // num_heads
        
        # Attention module parameters with fully fused QKV projection
        # Single QKV projection: outputs (num_heads + 2 * num_kv_heads) * head_dim
        qkv_proj_params = hidden_size * ((num_heads + 2 * num_kv_heads) * head_dim)
        o_proj_params = hidden_size * hidden_size  # Output projection
        attention_params = qkv_proj_params + o_proj_params
        
        # Feed-forward network parameters - SwiGLU always uses 3 matrices
        # SwiGLU uses gate, up, and down projections
        gate_params = hidden_size * intermediate_size
        up_params = hidden_size * intermediate_size
        down_params = intermediate_size * hidden_size
        ffn_params = gate_params + up_params + down_params
        
        # LayerNorm/RMSNorm parameters (2 per block: norm1 and norm2)
        # RMSNorm only has weight, no bias
        layernorm_params_per = hidden_size  # weight only (RMSNorm)
        layernorm_params = layernorm_params_per * 2  # norm1 + norm2
        
        # Total for single block
        single_block_params = attention_params + ffn_params + layernorm_params
        
        # All transformer layers
        num_layers = self.config['num_layers']
        all_blocks_params = single_block_params * num_layers
        final_norm_params = hidden_size  # Final RMSNorm (weight only)
        transformer_params = all_blocks_params + final_norm_params
        
        # Total unique parameters (excluding tied weights)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'embedding': embedding_params,
            'positional': positional_params,
            'attention_per_block': attention_params,
            'ffn_per_block': ffn_params,
            'layernorm_per_block': layernorm_params,
            'single_block': single_block_params,
            'all_blocks': all_blocks_params,
            'final_norm': final_norm_params,
            'transformer': transformer_params,
            'total': total_params,
            'trainable': trainable_params,
            'num_layers': num_layers
        }


def create_model(config_path: str = "config.json", model_size: str = None) -> TransformerLM:
    """
    Create a transformer language model from configuration file.
    Model will be automatically moved to GPU if available.
    
    Args:
        config_path: Path to the configuration JSON file
        model_size: Model size to use ("small" or "medium"). If None, uses default from config.
        
    Returns:
        Initialized TransformerLM model (on GPU if available)
    """
    # Check CUDA availability (required for training)
    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA is not available. Model will remain on CPU.")
        print("    For training, a GPU is required. Please ensure CUDA is properly installed.")
    else:
        # Log Flash Attention availability
        print(f"🚀 GPU detected: {torch.cuda.get_device_name()}")
        print("   Flash Attention will be used for optimal performance")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Handle model size selection
    if 'models' in config:
        # New multi-model config format
        available_sizes = list(config['models'].keys())
        
        # Use provided model_size or fall back to default
        if model_size is None:
            model_size = config.get('default_model_size', 'medium')
        
        # Validate model size
        if model_size not in available_sizes:
            raise ValueError(f"Invalid model size '{model_size}'. Available sizes: {available_sizes}")
        
        print(f"📊 Loading '{model_size}' model configuration")
        model_config = config['models'][model_size]
    else:
        # Backward compatibility with old single-model format
        if model_size is not None:
            print(f"⚠️  Warning: Config file uses old format. Ignoring model_size parameter.")
        model_config = config['model']
    tokenizer_config = config.get('tokenizer', {})
    rope_config = config.get('rope', {})
    architecture_features = config.get('architecture_features', {})
    
    # Create model with configuration, including optional architecture features
    model = TransformerLM(
        vocab_size=model_config['vocab_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        num_kv_heads=model_config.get('num_kv_heads', model_config['num_heads']),
        intermediate_size=model_config.get('intermediate_size'),  # Pass from config if specified
        max_position_embeddings=model_config['max_position_embeddings'],
        rope_theta=rope_config.get('theta', 10000.0),
        rope_scaling=rope_config.get('scaling_factor', 1.0),
        dropout=model_config['dropout'],
        attention_dropout=model_config['attention_dropout'],
        layer_norm_eps=model_config['layer_norm_eps'],
        initializer_range=model_config['initializer_range'],
        use_cache=model_config['use_cache'],
        pad_token_id=tokenizer_config.get('pad_token_id', 2),
        eos_token_id=tokenizer_config.get('eos_token_id', 2),
        # Architecture feature flags (with defaults for backward compatibility)
        use_scaled_residuals=architecture_features.get('use_scaled_residuals', True),
        use_gradient_checkpointing=architecture_features.get('use_gradient_checkpointing', False),
        use_gqa=architecture_features.get('use_gqa', True),
        use_flash_attention=architecture_features.get('use_flash_attention', True),
        tie_embeddings=architecture_features.get('tie_embeddings', True)
    )
    
    # Get parameter count for logging
    params = model.get_num_params()
    
    # Always compile for massive speedup (2-3x faster)
    model = torch.compile(model)
    print(f"Model created: {params['total']:,} parameters")

    return model


if __name__ == "__main__":
    import gc
    
    # Test model creation for all three configurations
    print("\n" + "="*70)
    print(" MODEL PARAMETER COMPARISON: SMALL vs MEDIUM vs LARGE")
    print("="*70)
    
    # Store models and their parameters for comparison
    model_configs = ["small", "medium", "large"]
    model_params = {}
    model_configs_saved = {}  # Store configurations separately
    
    for model_size in model_configs:
        print(f"\n{'='*70}")
        print(f" {model_size.upper()} MODEL CONFIGURATION")
        print('='*70)
        
        # Create model
        model = create_model(model_size=model_size)
        
        # Get and store parameter breakdown
        params = model.get_num_params()
        model_params[model_size] = params
        
        # Save configuration before deleting model
        config = model.config.copy()
        model_configs_saved[model_size] = config
        
        # Calculate attention head size and GQA ratio
        head_size = config['hidden_size'] // config['num_heads']
        gqa_ratio = config['num_heads'] // config['num_kv_heads']
        
        print(f"\n🎯 Architecture Configuration ({model_size}):")
        print(f"   Hidden Size: {config['hidden_size']} dimensions")
        print(f"   Head Size: {head_size} dimensions")
        print(f"   Query Heads: {config['num_heads']}")
        print(f"   KV Heads: {config['num_kv_heads']} (GQA ratio {gqa_ratio}:1)")
        print(f"   Num Layers: {config['num_layers']}")
        print(f"   Intermediate Size: {config['intermediate_size']}")
        print(f"   Vocab Size: {config['vocab_size']}")
        print(f"   Max Position: {config['max_position_embeddings']}")
        print(f"   RoPE Theta: {config.get('rope_theta', 10000.0)}")
        
        print("\n📦 Embeddings")
        print(f"   └── Token Embeddings:       {params['embedding']:>12,} params")
        print(f"   Total Embeddings:           {params['embedding']:>12,} params")
        print(f"   (RoPE is parameter-free)")
        
        print("\n🔧 Single Transformer Block")
        print(f"   ├── Attention Module:         {params['attention_per_block']:>11,} params")
        print(f"   ├── Feed-Forward Network:     {params['ffn_per_block']:>11,} params")
        print(f"   └── Layer Normalizations:     {params['layernorm_per_block']:>11,} params")
        print(f"   Total per block:              {params['single_block']:>11,} params")
        
        print(f"\n🏗️  All Transformer Layers ({params['num_layers']} blocks)")
        print(f"   ├── {params['num_layers']} × Transformer Blocks: {params['all_blocks']:>11,} params")
        print(f"   └── Final LayerNorm:          {params['final_norm']:>11,} params")
        print(f"   Total Transformer:            {params['transformer']:>11,} params")
        
        print("\n" + "-"*60)
        print(f"🎯 Total Model Parameters:      {params['total']:>11,} params")
        print(f"   Trainable Parameters:        {params['trainable']:>11,} params")
        print(f"   Memory (FP32):               {params['total'] * 4 / (1024**3):>11.2f} GB")
        print(f"   Memory (BF16):               {params['total'] * 2 / (1024**3):>11.2f} GB")
        print("-"*60)
        
        # Calculate percentages
        print("\n📊 Parameter Distribution")
        print(f"   Embeddings:      {params['embedding'] / params['total'] * 100:>5.1f}%")
        print(f"   Transformer:     {params['transformer'] / params['total'] * 100:>5.1f}%")
        print(f"     - Attention:   {(params['attention_per_block'] * params['num_layers']) / params['total'] * 100:>5.1f}% of total")
        print(f"     - FFN:         {(params['ffn_per_block'] * params['num_layers']) / params['total'] * 100:>5.1f}% of total")
        print(f"     - LayerNorms:  {(params['layernorm_per_block'] * params['num_layers'] + params['final_norm']) / params['total'] * 100:>5.1f}% of total")
        
        # Delete model to free memory before loading next one
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print(f"\n💾 Memory cleared after {model_size} model")
    
    # Final comparison between all three models
    print("\n" + "="*70)
    print(" FINAL COMPARISON: ALL THREE MODELS")
    print("="*70)
    
    small_total = model_params['small']['total']
    medium_total = model_params['medium']['total']
    large_total = model_params['large']['total']
    
    print(f"\n📈 Model Sizes:")
    print(f"   Small Model:  {small_total:>15,} parameters ({small_total / 1e6:>8.1f}M)")
    print(f"   Medium Model: {medium_total:>15,} parameters ({medium_total / 1e6:>8.1f}M)")
    print(f"   Large Model:  {large_total:>15,} parameters ({large_total / 1e9:>8.2f}B)")
    
    print(f"\n📊 Scale Factors:")
    print(f"   Medium vs Small: {medium_total / small_total:>8.2f}x larger")
    print(f"   Large vs Medium: {large_total / medium_total:>8.2f}x larger")
    print(f"   Large vs Small:  {large_total / small_total:>8.2f}x larger")
    
    # Get actual configuration values from saved configs
    small_config = model_configs_saved['small']
    medium_config = model_configs_saved['medium']
    large_config = model_configs_saved['large']
    
    print(f"\n🔢 Configuration Comparison:")
    print(f"   {'Parameter':<20} {'Small':>8} {'Medium':>8} {'Large':>8}")
    print(f"   {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    print(f"   {'Hidden Size':<20} {small_config['hidden_size']:>8} {medium_config['hidden_size']:>8} {large_config['hidden_size']:>8}")
    print(f"   {'Num Layers':<20} {small_config['num_layers']:>8} {medium_config['num_layers']:>8} {large_config['num_layers']:>8}")
    print(f"   {'Num Heads':<20} {small_config['num_heads']:>8} {medium_config['num_heads']:>8} {large_config['num_heads']:>8}")
    print(f"   {'KV Heads':<20} {small_config['num_kv_heads']:>8} {medium_config['num_kv_heads']:>8} {large_config['num_kv_heads']:>8}")
    print(f"   {'Head Dim':<20} {small_config['hidden_size']//small_config['num_heads']:>8} {medium_config['hidden_size']//medium_config['num_heads']:>8} {large_config['hidden_size']//large_config['num_heads']:>8}")
    print(f"   {'Max Positions':<20} {small_config['max_position_embeddings']:>8} {medium_config['max_position_embeddings']:>8} {large_config['max_position_embeddings']:>8}")
    print(f"   {'GQA Ratio':<20} {small_config['num_heads']//small_config['num_kv_heads']:>7}:1 {medium_config['num_heads']//medium_config['num_kv_heads']:>7}:1 {large_config['num_heads']//large_config['num_kv_heads']:>7}:1")
    
    print(f"\n💾 Memory Requirements (Weights Only):")
    print(f"   {'Model':<20} {'FP32':>12} {'BF16/FP16':>12}")
    print(f"   {'-'*20} {'-'*12} {'-'*12}")
    print(f"   {'Small':<20} {small_total * 4 / (1024**3):>10.2f} GB {small_total * 2 / (1024**3):>10.2f} GB")
    print(f"   {'Medium':<20} {medium_total * 4 / (1024**3):>10.2f} GB {medium_total * 2 / (1024**3):>10.2f} GB")
    print(f"   {'Large':<20} {large_total * 4 / (1024**3):>10.2f} GB {large_total * 2 / (1024**3):>10.2f} GB")
    
    print(f"\n⚡ Training Memory Estimates (BF16 with Adam):")
    # Adam optimizer maintains 2 states per parameter (momentum + variance)
    # Total = weights (2 bytes) + gradients (2 bytes) + optimizer states (2*2 bytes) = 8 bytes per param
    optimizer_multiplier = 4  # 8 bytes / 2 bytes base = 4x
    print(f"   Small:  ~{small_total * 2 * optimizer_multiplier / (1024**3):.1f} GB (weights + gradients + optimizer)")
    print(f"   Medium: ~{medium_total * 2 * optimizer_multiplier / (1024**3):.1f} GB (weights + gradients + optimizer)")
    print(f"   Large:  ~{large_total * 2 * optimizer_multiplier / (1024**3):.1f} GB (weights + gradients + optimizer)")
    print(f"\n   Note: Add ~2-8 GB for activations depending on batch size")
    
    print("\n" + "="*70)
