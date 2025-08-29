#!/usr/bin/env python3
"""
Transformer Language Model Implementation

This module defines a standard transformer architecture for language modeling,
using native PyTorch modules for efficiency and simplicity.
"""

import json
import math
import os
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# Import our custom modules
from utils.rope import RotaryEmbedding
from utils.activations import SwiGLU

# Set environment variable for optimal memory allocation on GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Enable TF32 for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

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
    Optimized Multi-Head Attention with Grouped Query Attention (GQA) support.
    Uses RoPE for position encoding and forced Flash Attention for maximum performance.
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
        rope_scaling: float = 1.0
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Ensure num_heads is divisible by num_kv_heads for even grouping
        assert self.num_heads % self.num_kv_heads == 0, \
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        self.num_groups = self.num_heads // self.num_kv_heads
        
        # Separate projections for GQA
        # Q projection uses all heads, KV projections use fewer heads
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        
        # Mark attention projections for specialized initialization
        self.q_proj._is_attention_qkv = True
        self.k_proj._is_attention_qkv = True
        self.v_proj._is_attention_qkv = True
        
        # Output projection
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
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
        is_causal: bool = True
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute Q, K, V projections separately for GQA
        # Q: (batch, seq_len, num_heads * head_dim)
        q = self.q_proj(hidden_states)
        # K, V: (batch, seq_len, num_kv_heads * head_dim)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape Q to (batch, num_heads, seq_len, head_dim)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        
        # Reshape K, V to (batch, num_kv_heads, seq_len, head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        k = k.transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.transpose(1, 2)
        
        # Apply RoPE to Q and K
        q, k = self.rotary_emb(q, k)
        
        # Expand K and V to match Q heads for GQA
        if self.num_kv_heads != self.num_heads:
            # Repeat KV heads to match Q heads
            # Shape: (batch, num_kv_heads, seq_len, head_dim) -> (batch, num_heads, seq_len, head_dim)
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)
        
        # Use Flash Attention backend for optimal performance when available
        if torch.cuda.is_available():
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
            # CPU fallback - let PyTorch choose the best backend
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
        
        return output

class FastTransformerDecoderLayer(nn.Module):
    """
    Optimized Transformer Decoder Layer using FastMultiHeadAttention with RoPE.
    Uses SwiGLU activation and pre-norm architecture for better training stability.
    Includes residual scaling and gradient checkpointing support.
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
        use_rms_norm: bool = True,
        use_swiglu: bool = True,
        max_position_embeddings: int = 512,
        rope_theta: float = 10000.0,
        rope_scaling: float = 1.0,
        layer_idx: int = 0,
        num_layers: int = 1,
        use_scaled_residuals: bool = True,
        use_gradient_checkpointing: bool = False
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
            bias=bias,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling
        )
        
        # Feed-forward network - use SwiGLU or standard FFN
        self.use_swiglu = use_swiglu
        if use_swiglu:
            self.ffn = SwiGLU(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                bias=bias,
                dropout=dropout
            )
        else:
            # Fallback to standard FFN
            self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
            self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=bias)
        
        # Use RMSNorm or LayerNorm based on configuration
        norm_class = nn.RMSNorm if use_rms_norm else nn.LayerNorm
        self.norm1 = norm_class(hidden_size, eps=layer_norm_eps)
        self.norm2 = norm_class(hidden_size, eps=layer_norm_eps)
        
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
        is_causal: bool = True
    ) -> torch.Tensor:
        """Attention sub-block for gradient checkpointing."""
        # Pre-norm architecture: LayerNorm before attention
        hidden_states = self.norm1(hidden_states)
        
        # Self-attention
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            is_causal=is_causal
        )
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states
    
    def _ffn_block(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Feed-forward sub-block for gradient checkpointing."""
        # Pre-norm
        hidden_states = self.norm2(hidden_states)
        
        if self.use_swiglu:
            # Use SwiGLU activation
            hidden_states = self.ffn(hidden_states)
        else:
            # Standard FFN with GELU activation
            hidden_states = self.fc1(hidden_states)
            hidden_states = F.gelu(hidden_states)
            hidden_states = self.fc2(hidden_states)
            hidden_states = self.dropout(hidden_states)
        
        return hidden_states
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        # Gradient checkpointing logic
        if self.use_gradient_checkpointing and self.training:
            # Checkpoint attention block
            residual = hidden_states
            attn_output = torch.utils.checkpoint.checkpoint(
                self._attention_block,
                hidden_states,
                attention_mask,
                is_causal,
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
        else:
            # Normal forward pass without checkpointing
            # Attention block
            residual = hidden_states
            attn_output = self._attention_block(hidden_states, attention_mask, is_causal)
            
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
        
        return hidden_states

class FastTransformerDecoder(nn.Module):
    """
    Stack of optimized Transformer Decoder layers with RoPE and SwiGLU.
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
        use_rms_norm: bool = True,
        use_swiglu: bool = True,
        max_position_embeddings: int = 512,
        rope_theta: float = 10000.0,
        rope_scaling: float = 1.0,
        use_scaled_residuals: bool = True,
        use_gradient_checkpointing: bool = False
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
                bias=bias,
                use_rms_norm=use_rms_norm,
                use_swiglu=use_swiglu,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                layer_idx=i,
                num_layers=num_layers,
                use_scaled_residuals=use_scaled_residuals,
                use_gradient_checkpointing=use_gradient_checkpointing
            )
            for i in range(num_layers)
        ])
        
        # Final layer norm - use RMSNorm or LayerNorm based on config
        norm_class = nn.RMSNorm if use_rms_norm else nn.LayerNorm
        self.norm = norm_class(hidden_size, eps=layer_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        # Process through each layer
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                is_causal=is_causal
            )
        
        # Apply final layer norm
        hidden_states = self.norm(hidden_states)
        
        return hidden_states

class TransformerLM(nn.Module):
    """Transformer Language Model with RoPE and SwiGLU."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        num_kv_heads: Optional[int] = None,
        intermediate_size: Optional[int] = None,  # Allow explicit intermediate size
        use_rms_norm: bool = True,
        use_swiglu: bool = True,
        max_position_embeddings: int = 512,
        rope_theta: float = 10000.0,
        rope_scaling: float = 1.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        pad_token_id: int = 32000,  # Changed from 2 (EOS) to 32000 (dedicated padding)
        eos_token_id: int = 2,
        use_scaled_residuals: bool = True,
        use_gradient_checkpointing: bool = False
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
            'use_rms_norm': use_rms_norm,
            'use_swiglu': use_swiglu,
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
            'use_gradient_checkpointing': use_gradient_checkpointing
        }
        
        # Token embeddings only (RoPE handles position encoding)
        self.embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        
        # Embedding dropout
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Use optimized FastTransformerDecoder with GQA, RoPE, SwiGLU, and RMSNorm
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
            use_rms_norm=use_rms_norm,
            use_swiglu=use_swiglu,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            use_scaled_residuals=use_scaled_residuals,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        
        # Language modeling head (always tied with input embeddings)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Always tie weights between input and output embeddings
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
            # Get the parent module name for context
            parent_name = module.__class__.__name__ if hasattr(module, '__class__') else ''
            
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
            # Gate and up projections: Xavier uniform
            if hasattr(module, 'gate_proj'):
                nn.init.xavier_uniform_(module.gate_proj.weight, gain=1.0)
                if module.gate_proj.bias is not None:
                    module.gate_proj.bias.data.zero_()
                    
            if hasattr(module, 'up_proj'):
                nn.init.xavier_uniform_(module.up_proj.weight, gain=1.0)
                if module.up_proj.bias is not None:
                    module.up_proj.bias.data.zero_()
                    
            # Down projection: scaled by depth
            if hasattr(module, 'down_proj'):
                std = self.config['initializer_range'] * depth_scale
                module.down_proj.weight.data.normal_(mean=0.0, std=std)
                if module.down_proj.bias is not None:
                    module.down_proj.bias.data.zero_()
    
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
        return_dict: bool = True
    ) -> Union[dict, tuple]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask for padding (batch_size, seq_len)
            labels: Target token IDs for language modeling loss
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary with 'logits' and 'loss' (if labels provided)
        """
        # Get token embeddings (RoPE handles position encoding internally)
        hidden_states = self.embeddings(input_ids)
        hidden_states = self.embedding_dropout(hidden_states)
        
        # Pass through optimized transformer decoder with RoPE and SwiGLU
        # The FastTransformerDecoder handles causal masking and position encoding internally
        hidden_states = self.transformer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            is_causal=True  # Always use causal masking for language modeling
        )
        
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
            
            # Also mask out actual padding tokens (ID 32000 or from config)
            pad_token_id = self.config.get('pad_token_id', 32000)
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
            return {
                'logits': logits,
                'loss': loss,
                'hidden_states': hidden_states
            }
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
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting token IDs (batch_size, seq_len)
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID
            
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
        
        # Use bfloat16 autocast for generation (matches training precision)
        # This ensures Flash Attention and other optimized kernels work correctly
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            while generated.shape[1] < max_length:
                # Forward pass - get logits for the last position
                outputs = self.forward(generated, return_dict=True)
                assert isinstance(outputs, dict)  # Type assertion for type checker
                next_token_logits = outputs['logits'][:, -1, :]
                
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
        use_swiglu = self.config.get('use_swiglu', True)
        
        # Attention module parameters with GQA
        q_proj_params = hidden_size * (num_heads * head_dim)  # Q projection
        k_proj_params = hidden_size * (num_kv_heads * head_dim)  # K projection (smaller with GQA)
        v_proj_params = hidden_size * (num_kv_heads * head_dim)  # V projection (smaller with GQA)
        o_proj_params = hidden_size * hidden_size  # Output projection
        attention_params = q_proj_params + k_proj_params + v_proj_params + o_proj_params
        
        # Feed-forward network parameters (SwiGLU has 3 matrices instead of 2)
        if use_swiglu:
            # SwiGLU uses gate, up, and down projections
            gate_params = hidden_size * intermediate_size
            up_params = hidden_size * intermediate_size
            down_params = intermediate_size * hidden_size
            ffn_params = gate_params + up_params + down_params
        else:
            # Standard FFN
            fc1_params = hidden_size * intermediate_size
            fc2_params = intermediate_size * hidden_size
            ffn_params = fc1_params + fc2_params
        
        # LayerNorm parameters (2 per block: norm1 and norm2)
        layernorm_params_per = hidden_size * 2  # weight + bias
        layernorm_params = layernorm_params_per * 2  # norm1 + norm2
        
        # Total for single block
        single_block_params = attention_params + ffn_params + layernorm_params
        
        # All transformer layers
        num_layers = self.config['num_layers']
        all_blocks_params = single_block_params * num_layers
        final_norm_params = hidden_size * 2  # Final LayerNorm
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
    
    # Create model with configuration, including GQA, RoPE, SwiGLU, and Flash Attention
    # Use backward-compatible defaults for old configs
    model = TransformerLM(
        vocab_size=model_config['vocab_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        num_kv_heads=model_config.get('num_kv_heads', model_config['num_heads']),
        use_rms_norm=model_config.get('use_rms_norm', True),
        use_swiglu=model_config.get('use_swiglu', True),  # Default to SwiGLU
        max_position_embeddings=model_config['max_position_embeddings'],
        rope_theta=rope_config.get('theta', 10000.0),
        rope_scaling=rope_config.get('scaling_factor', 1.0),
        dropout=model_config['dropout'],
        attention_dropout=model_config['attention_dropout'],
        layer_norm_eps=model_config['layer_norm_eps'],
        initializer_range=model_config['initializer_range'],
        use_cache=model_config['use_cache'],
        pad_token_id=tokenizer_config.get('pad_token_id', 32000),
        eos_token_id=tokenizer_config.get('eos_token_id', 2)
    )
    
    # Get parameter count for logging
    params = model.get_num_params()
    
    model = torch.compile(model)
    print(f"Model created: {params['total']:,} parameters")

    return model


if __name__ == "__main__":
    import sys
    
    # Test model creation - accept model size from command line
    model_size = sys.argv[1] if len(sys.argv) > 1 else None
    if model_size:
        print(f"Testing with model size: {model_size}")
    model = create_model(model_size=model_size)
    
    # Display detailed parameter breakdown
    params = model.get_num_params()
    print("\n" + "="*60)
    print("Model Parameter Breakdown")
    print("="*60)
    
    # Calculate attention head size and GQA ratio
    config = model.config
    head_size = config['hidden_size'] // config['num_heads']
    gqa_ratio = config['num_heads'] // config['num_kv_heads']
    print(f"\n🎯 Architecture Configuration:")
    print(f"   Head Size: {head_size} dimensions")
    print(f"   Query Heads: {config['num_heads']}")
    print(f"   KV Heads: {config['num_kv_heads']} (GQA ratio {gqa_ratio}:1)")
    print(f"   Using RMSNorm: {config['use_rms_norm']}")
    print(f"   Using SwiGLU: {config.get('use_swiglu', True)}")
    print(f"   Using RoPE: Yes (θ={config.get('rope_theta', 10000.0)})")
    print(f"   Flash Attention: Enabled (auto-detect)")
    
    print("\n📦 Embeddings")
    print(f"   └── Token Embeddings:       {params['embedding']:>12,} params")
    print(f"   Total Embeddings:           {params['embedding']:>12,} params")
    print(f"   (RoPE is parameter-free)")
    
    print("\n🔧 Single Transformer Block")
    print(f"   ├── Attention (QKV + O_proj): {params['attention_per_block']:>11,} params")
    ffn_type = "SwiGLU (3 matrices)" if config.get('use_swiglu', True) else "Standard FFN"
    print(f"   ├── {ffn_type:24} {params['ffn_per_block']:>11,} params")
    norm_type = "RMSNorms" if config.get('use_rms_norm', True) else "LayerNorms"
    print(f"   └── {norm_type} (2x):            {params['layernorm_per_block']:>11,} params")
    print(f"   Total per block:              {params['single_block']:>11,} params")
    
    print(f"\n🏗️  All Transformer Layers ({params['num_layers']} blocks)")
    print(f"   ├── {params['num_layers']} × Transformer Blocks: {params['all_blocks']:>11,} params")
    print(f"   └── Final LayerNorm:          {params['final_norm']:>11,} params")
    print(f"   Total Transformer:            {params['transformer']:>11,} params")
    
    print("\n" + "="*60)
    print(f"🎯 Total Model Parameters:      {params['total']:>11,} params")
    print(f"   Trainable Parameters:        {params['trainable']:>11,} params")
    print("="*60)
    
    # Calculate percentages
    print("\n📊 Parameter Distribution")
    print(f"   Embeddings:      {params['embedding'] / params['total'] * 100:>5.1f}%")
    print(f"   Transformer:     {params['transformer'] / params['total'] * 100:>5.1f}%")
    print(f"     - Attention:   {(params['attention_per_block'] * params['num_layers']) / params['total'] * 100:>5.1f}% of total")
    ffn_label = "SwiGLU" if config.get('use_swiglu', True) else "FFN"
    print(f"     - {ffn_label:10} {(params['ffn_per_block'] * params['num_layers']) / params['total'] * 100:>5.1f}% of total")
    print("\n" + "="*60)
