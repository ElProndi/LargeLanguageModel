#!/usr/bin/env python3
"""
Transformer Language Model Implementation

This module defines a standard transformer architecture for language modeling,
using native PyTorch modules for efficiency and simplicity.
"""

import json
import math
import time
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding
from .activations import SwiGLU

# Enable TF32 and Flash Attention
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.enable_flash_sdp(True)

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention.
    Uses RoPE for position encoding with automatic backend optimization.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        rope_scaling: float = 1.0
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Fully fused QKV projection: single GEMM producing Q, K, and V
        # Output size is 3 * hidden_size for multi-head attention
        self.qkv_proj = nn.Linear(
            hidden_size, 
            3 * hidden_size, 
            bias=False
        )
        
        # Mark attention projection for specialized initialization
        self.qkv_proj._is_attention_qkv = True
        
        # Output projection
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj._is_attention_output = True
        
        self.dropout = dropout
        
        # Initialize RoPE
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            theta=rope_theta,
            scaling_factor=rope_scaling,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
    
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Single fused QKV projection - reduces 3 GEMMs to 1
        # Output: (batch, seq_len, 3 * hidden_size)
        qkv = self.qkv_proj(hidden_states)
        
        # Efficient QKV reshape: single view + permute instead of chunk + 6 ops
        # Reshape from (batch, seq_len, 3 * hidden_size) to (batch, seq_len, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        # Permute to (3, batch, num_heads, seq_len, head_dim) and unpack
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        # Apply RoPE to Q and K
        q, k = self.rotary_emb(q, k, position_offset=0)
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,  # Always use causal masking for language modeling
            scale=self.scale
        )
        
        # Reshape back to (batch, seq_len, hidden_size)
        # More efficient: single operation instead of transpose + contiguous + reshape
        attn_output = attn_output.transpose(1, 2).flatten(2)
        output = self.o_proj(attn_output)
        
        return output

class TransformerLayer(nn.Module):
    """
    Transformer Decoder Layer using MultiHeadAttention with RoPE.
    Uses SwiGLU activation and RMSNorm with pre-norm architecture.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        rope_scaling: float = 1.0,
        layer_idx: int = 0,
        num_layers: int = 1
    ):
        super().__init__()
        
        if intermediate_size is None:
            intermediate_size = hidden_size * 4
        
        # Store layer index for layer-dependent initialization
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        
        # Self-attention components with RoPE
        self.self_attn = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=attention_dropout,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling
        )
        # Mark attention module with layer index
        self.self_attn._layer_idx = layer_idx
        self.self_attn._num_layers = num_layers
        
        # Feed-forward network - always use SwiGLU for better gradient flow
        self.ffn = SwiGLU(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=False,  # No bias for better kernel fusion and memory savings
            dropout=dropout
        )
        # Mark FFN module with layer index
        self.ffn._layer_idx = layer_idx
        self.ffn._num_layers = num_layers
        
        # Always use RMSNorm for better efficiency
        self.norm1 = nn.RMSNorm(hidden_size, eps=layer_norm_eps)
        self.norm2 = nn.RMSNorm(hidden_size, eps=layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # Attention block with pre-norm
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        
        # Self-attention
        hidden_states = self.self_attn(
            hidden_states
        )
        
        hidden_states = self.dropout(hidden_states)
        
        # Apply residual connection
        hidden_states = residual + hidden_states
        
        # FFN block with pre-norm
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        
        # Feed-forward network with SwiGLU
        hidden_states = self.ffn(hidden_states)
        
        # Apply residual connection
        hidden_states = residual + hidden_states
        
        return hidden_states

class TransformerDecoder(nn.Module):
    """
    Stack of optimized Transformer Decoder layers with RoPE, SwiGLU, and RMSNorm.
    Eliminates unnecessary operations from nn.TransformerDecoder.
    """
    
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        rope_scaling: float = 1.0
    ):
        super().__init__()
        
        if intermediate_size is None:
            intermediate_size = hidden_size * 4
        
        # Stack of decoder layers with RoPE, SwiGLU, and RMSNorm support
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                attention_dropout=attention_dropout,
                layer_norm_eps=layer_norm_eps,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                layer_idx=i,
                num_layers=num_layers
            )
            for i in range(num_layers)
        ])
        
        # Final layer norm - always use RMSNorm for efficiency
        self.norm = nn.RMSNorm(hidden_size, eps=layer_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # Process through each layer
        for layer in self.layers:
            hidden_states = layer(
                hidden_states
            )
        
        # Apply final layer norm
        hidden_states = self.norm(hidden_states)
        
        return hidden_states

class TransformerLM(nn.Module):
    """Transformer Language Model with RoPE, SwiGLU, and RMSNorm."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        intermediate_size: Optional[int] = None,  # Allow explicit intermediate size
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        rope_scaling: float = 1.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        pad_token_id: int = 2,  # Same as EOS token (CodeLlama convention)
        eos_token_id: int = 2,
        tie_embeddings: bool = True
    ):
        super().__init__()
        
        # Calculate intermediate size if not provided
        if intermediate_size is None:
            intermediate_size = hidden_size * 3
        
        self.config = {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'intermediate_size': intermediate_size,
            'max_position_embeddings': max_position_embeddings,
            'rope_theta': rope_theta,
            'rope_scaling': rope_scaling,
            'dropout': dropout,
            'attention_dropout': attention_dropout,
            'layer_norm_eps': layer_norm_eps,
            'initializer_range': initializer_range,
            'pad_token_id': pad_token_id,
            'eos_token_id': eos_token_id,
            'tie_embeddings': tie_embeddings
        }
        
        # Token embeddings only (RoPE handles position encoding)
        self.embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        
        # Embedding dropout
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Use optimized TransformerDecoder with configurable features
        self.transformer = TransformerDecoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            dropout=dropout,
            attention_dropout=attention_dropout,
            layer_norm_eps=layer_norm_eps,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling
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
        depth_scale = 1.0 / math.sqrt(num_layers) if num_layers > 1 else 1.0
        
        # Get layer-specific scaling if module has layer index
        layer_scale = 1.0
        if hasattr(module, '_layer_idx'):
            # Progressive scaling: early layers get larger variance, late layers get smaller
            # This helps early layers learn features while late layers fine-tune
            layer_idx = module._layer_idx
            layer_scale = 1.0 - 0.1 * (layer_idx / num_layers)
        
        if isinstance(module, nn.Linear):
            # Check if this is part of attention mechanism
            if hasattr(module, '_is_attention_qkv'):
                # Query, Key, Value projections - improved attention-aware initialization
                # Account for head dimension and attention score scaling
                head_dim = self.config['hidden_size'] // self.config['num_heads']
                
                # Apply layer-dependent scaling to attention
                parent = module
                while parent is not None:
                    if hasattr(parent, '_layer_idx'):
                        layer_idx = parent._layer_idx
                        layer_scale = 1.0 - 0.1 * (layer_idx / num_layers)
                        break
                    parent = getattr(parent, '_parent', None)
                
                # Scale initialization to account for attention score computation:
                # - Queries and keys will be dot-producted and scaled by 1/sqrt(head_dim)
                # - We want attention scores to have reasonable magnitude before softmax
                # - Use smaller std for Q,K to prevent attention saturation, normal std for V
                
                # The QKV projection outputs [Q, K, V] concatenated
                # So we initialize the entire weight matrix, but conceptually:
                # - Q and K portions: smaller variance to control attention score magnitude  
                # - V portion: standard variance since it doesn't affect attention scores
                std_qk = self.config['initializer_range'] * layer_scale / math.sqrt(head_dim / 64.0)  # Scale relative to typical head_dim=64
                std_v = self.config['initializer_range'] * layer_scale
                
                # Initialize Q,K portions with smaller variance, V with standard variance
                weight = module.weight.data
                hidden_size = self.config['hidden_size']
                
                # Split weight into Q, K, V portions (each is hidden_size x hidden_size)
                q_weight = weight[:hidden_size, :]  # First 1/3
                k_weight = weight[hidden_size:2*hidden_size, :]  # Second 1/3  
                v_weight = weight[2*hidden_size:, :]  # Last 1/3
                
                # Initialize each portion separately
                q_weight.normal_(mean=0.0, std=std_qk)
                k_weight.normal_(mean=0.0, std=std_qk)
                v_weight.normal_(mean=0.0, std=std_v)
                
            elif hasattr(module, '_is_attention_output'):
                # Attention output projection - scale by depth and layer
                parent = module
                while parent is not None:
                    if hasattr(parent, '_layer_idx'):
                        layer_idx = parent._layer_idx
                        layer_scale = 1.0 - 0.1 * (layer_idx / num_layers)
                        break
                    parent = getattr(parent, '_parent', None)
                    
                std = self.config['initializer_range'] * depth_scale * layer_scale
                module.weight.data.normal_(mean=0.0, std=std)
                
            elif isinstance(module.weight, nn.Parameter) and module.out_features > module.in_features * 2:
                # Likely FFN up/gate projection - use He initialization for activation layers
                # He initialization is better for layers with non-linear activations
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                
            elif isinstance(module.weight, nn.Parameter) and module.in_features > module.out_features * 2:
                # Likely FFN down projection - scale by depth and layer
                std = self.config['initializer_range'] * depth_scale * layer_scale
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
            # Special initialization for SwiGLU components with layer-dependent scaling
            layer_scale = 1.0
            if hasattr(module, '_layer_idx'):
                layer_idx = module._layer_idx
                layer_scale = 1.0 - 0.1 * (layer_idx / num_layers)
            
            # Fused gate+up projection: differentiated initialization
            if hasattr(module, 'w_fused'):
                # The fused weight contains both gate and up projections
                # Split conceptually for different initialization
                weight = module.w_fused.weight.data
                intermediate_size = weight.shape[0] // 2
                
                # Gate projection (first half) - smaller variance for stability
                gate_weight = weight[:intermediate_size, :]
                gate_std = self.config['initializer_range'] * layer_scale * 0.8
                gate_weight.normal_(mean=0.0, std=gate_std)
                
                # Up projection (second half) - He initialization for activation
                up_weight = weight[intermediate_size:, :]
                nn.init.kaiming_uniform_(up_weight, mode='fan_in', nonlinearity='relu')
                
                if module.w_fused.bias is not None:
                    module.w_fused.bias.data.zero_()
                    
            # Down projection: scaled by depth and layer
            if hasattr(module, 'w_down'):
                std = self.config['initializer_range'] * depth_scale * layer_scale
                module.w_down.weight.data.normal_(mean=0.0, std=std)
                if module.w_down.bias is not None:
                    module.w_down.bias.data.zero_()
    
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[dict, tuple]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            labels: Target token IDs for language modeling loss
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary with 'logits', 'loss' (if labels provided), and 'hidden_states'
        """
        # Get token embeddings (RoPE handles position encoding internally)
        hidden_states = self.embeddings(input_ids)
        hidden_states = self.embedding_dropout(hidden_states)
        
        # Pass through optimized transformer decoder with RoPE and SwiGLU
        # The TransformerDecoder handles causal masking and position encoding internally
        hidden_states = self.transformer(
            hidden_states=hidden_states
        )
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute cross-entropy loss on the entire batch
            # No masking - calculate loss on all tokens
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config['vocab_size']),
                shift_labels.view(-1)
            )
        
        if return_dict:
            output = {
                'logits': logits,
                'loss': loss,
                'hidden_states': hidden_states
            }
            return output
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
                # Forward pass - process full sequence each time
                outputs = self.forward(
                    generated, 
                    return_dict=True
                )
                assert isinstance(outputs, dict)  # Type assertion for type checker
                
                # Extract logits for the last position
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
    
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None
    ):
        """
        Generate text autoregressively with streaming output.
        Yields tokens one at a time as they're generated.
        
        Args:
            input_ids: Starting token IDs (batch_size, seq_len)
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID
            
        Yields:
            Individual token IDs as they're generated
            Final yield is a dict with generation statistics
        """
        self.eval()
        pad_token_id = pad_token_id or self.config['pad_token_id']
        eos_token_id = eos_token_id or self.config.get('eos_token_id', 2)
        
        # Initialize with input
        generated = input_ids
        batch_size = input_ids.shape[0]
        
        # Track which sequences are done
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        
        # Track statistics
        tokens_generated = 0
        start_time = time.time()
        
        # Use bfloat16 autocast for generation (matches training precision)
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            while generated.shape[1] < max_length:
                # Forward pass - process full sequence each time
                outputs = self.forward(
                    generated, 
                    return_dict=True
                )
                assert isinstance(outputs, dict)
                
                # Extract logits for the last position
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
                
                # Yield the new token (only for batch index 0 for simplicity)
                if unfinished_sequences[0] == 1:  # Only yield if sequence 0 is not finished
                    yield next_tokens[0, 0].item()
                    tokens_generated += 1
                
                # Update which sequences are finished
                unfinished_sequences = unfinished_sequences * (next_tokens.squeeze(-1) != eos_token_id).long()
                
                # Stop if all sequences are done
                if unfinished_sequences.sum() == 0:
                    break
        
        # Final yield with statistics
        generation_time = time.time() - start_time
        yield {
            'complete': True,
            'tokens_generated': tokens_generated,
            'generation_time': generation_time,
            'tokens_per_sec': tokens_generated / generation_time if generation_time > 0 else 0,
            'full_sequence': generated
        }
    
    
    def get_num_params(self) -> dict:
        """Get detailed breakdown of model parameters."""
        # Token embeddings (shared with output layer due to tying)
        embedding_params = self.embeddings.weight.numel()
        
        # No positional embeddings anymore (RoPE is parameter-free)
        positional_params = 0
        
        # Calculate parameters for a single transformer block
        hidden_size = self.config['hidden_size']
        intermediate_size = self.config['intermediate_size']
        
        # Attention module parameters with fully fused QKV projection
        # Single QKV projection: outputs 3 * hidden_size
        qkv_proj_params = hidden_size * (3 * hidden_size)
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
    # Log GPU availability
    print(f"🚀 GPU detected: {torch.cuda.get_device_name()}")
    print("   Optimized attention backend will be used automatically")
    
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
        intermediate_size=model_config.get('intermediate_size'),  # Pass from config if specified
        max_position_embeddings=model_config['max_position_embeddings'],
        rope_theta=rope_config.get('theta', 10000.0),
        rope_scaling=rope_config.get('scaling_factor', 1.0),
        dropout=model_config['dropout'],
        attention_dropout=model_config['attention_dropout'],
        layer_norm_eps=model_config['layer_norm_eps'],
        initializer_range=model_config['initializer_range'],
        pad_token_id=tokenizer_config.get('pad_token_id', 2),
        eos_token_id=tokenizer_config.get('eos_token_id', 2),
        # Architecture feature flags (with defaults for backward compatibility)
        tie_embeddings=architecture_features.get('tie_embeddings', True)
    )
    
    # Get parameter count for logging
    params = model.get_num_params()
    
    # Always compile
    #model = torch.compile(model)
    model = torch.compile(model,mode="max-autotune-no-cudagraphs", fullgraph=True, dynamic=False)
    print(f"Model created: {params['total']:,} parameters")

    return model