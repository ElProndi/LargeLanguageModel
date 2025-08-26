#!/usr/bin/env python3
"""
Transformer Language Model Implementation

This module defines a standard transformer architecture for language modeling,
using native PyTorch modules for efficiency and simplicity.
"""

import json
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

class FastMultiHeadAttention(nn.Module):
    """
    Optimized Multi-Head Attention using packed projections and scaled_dot_product_attention.
    This implementation is significantly faster than nn.MultiheadAttention.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Packed projection: Q, K, V in a single Linear layer
        # This reduces memory bandwidth by 3x compared to separate projections
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        
        # Output projection
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        self.dropout = dropout
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Packed projection and split into Q, K, V
        # Shape: (batch, seq_len, 3 * hidden_size)
        qkv = self.qkv_proj(hidden_states)
        
        # Reshape to separate Q, K, V and heads
        # Shape: (batch, seq_len, 3, num_heads, head_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        
        # Permute to (3, batch, num_heads, seq_len, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        # Split into separate Q, K, V tensors
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Use PyTorch's optimized scaled_dot_product_attention
        # This automatically selects the best backend (FlashAttention, Memory-Efficient, or Math)
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
    Optimized Transformer Decoder Layer using FastMultiHeadAttention.
    Uses pre-norm architecture for better training stability.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        bias: bool = True
    ):
        super().__init__()
        
        # Self-attention components
        self.self_attn = FastMultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=attention_dropout,
            bias=bias
        )
        
        # Feed-forward network
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=bias)
        
        # Layer norms (pre-norm architecture)
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        # Pre-norm architecture: LayerNorm before attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        
        # Self-attention
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            is_causal=is_causal
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # Feed-forward network with pre-norm
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        
        # FFN with GELU activation
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class FastTransformerDecoder(nn.Module):
    """
    Stack of optimized Transformer Decoder layers.
    Eliminates unnecessary operations from nn.TransformerDecoder.
    """
    
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        bias: bool = True
    ):
        super().__init__()
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            FastTransformerDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                attention_dropout=attention_dropout,
                layer_norm_eps=layer_norm_eps,
                bias=bias
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
    
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
    """Transformer Language Model using native PyTorch modules."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        max_position_embeddings: int = 512,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        pad_token_id: int = 3
    ):
        super().__init__()
        
        # Calculate intermediate size as 4x hidden size (standard transformer pattern)
        intermediate_size = hidden_size * 4
        
        self.config = {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'intermediate_size': intermediate_size,
            'max_position_embeddings': max_position_embeddings,
            'dropout': dropout,
            'attention_dropout': attention_dropout,
            'layer_norm_eps': layer_norm_eps,
            'initializer_range': initializer_range,
            'use_cache': use_cache,
            'pad_token_id': pad_token_id
        }
        
        # Token embeddings
        self.embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        
        # Positional embeddings (learned, like GPT-2)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Embedding dropout
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Use optimized FastTransformerDecoder instead of nn.TransformerDecoder
        self.transformer = FastTransformerDecoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            dropout=dropout,
            attention_dropout=attention_dropout,
            layer_norm_eps=layer_norm_eps,
            bias=True
        )
        
        # Language modeling head (always tied with input embeddings)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Always tie weights between input and output embeddings
        self.lm_head.weight = self.embeddings.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Register buffer for position IDs
        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
            persistent=False
        )
        
        # Always move model to GPU if available
        if torch.cuda.is_available():
            self.cuda()
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config['initializer_range'])
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config['initializer_range'])
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[dict, tuple]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask for padding (batch_size, seq_len)
            labels: Target token IDs for language modeling loss
            position_ids: Position IDs for positional embeddings
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary with 'logits' and 'loss' (if labels provided)
        """
        seq_len = input_ids.shape[1]
        
        # Get position IDs if not provided
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_len]  # type: ignore
        
        # Get token and position embeddings
        token_embeddings = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        # Combine embeddings
        hidden_states = token_embeddings + position_embeddings
        hidden_states = self.embedding_dropout(hidden_states)
        
        # Pass through optimized transformer decoder
        # The FastTransformerDecoder handles causal masking internally
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
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config['vocab_size']),
                shift_labels.view(-1),
                ignore_index=self.config['pad_token_id']
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
        eos_token_id: int = 1,
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
        
        # Initialize with input
        generated = input_ids
        batch_size = input_ids.shape[0]
        
        # Track which sequences are done
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        
        with torch.no_grad():
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
        
        # Positional embeddings
        positional_params = self.position_embeddings.weight.numel()
        
        # Transformer layers
        transformer_params = sum(p.numel() for name, p in self.named_parameters() 
                               if 'transformer' in name)
        
        # Total unique parameters (excluding tied weights)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'embedding': embedding_params,
            'positional': positional_params,
            'transformer': transformer_params,
            'total': total_params,
            'trainable': trainable_params
        }


def create_model(config_path: str = "config.json") -> TransformerLM:
    """
    Create a transformer language model from configuration file.
    Model will be automatically moved to GPU if available.
    
    Args:
        config_path: Path to the configuration JSON file
        
    Returns:
        Initialized TransformerLM model (on GPU if available)
    """
    # Check CUDA availability (required for training)
    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA is not available. Model will remain on CPU.")
        print("    For training, a GPU is required. Please ensure CUDA is properly installed.")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract model configuration
    model_config = config['model']
    tokenizer_config = config.get('tokenizer', {})
    
    # Create model with configuration
    model = TransformerLM(
        vocab_size=model_config['vocab_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        max_position_embeddings=model_config['max_position_embeddings'],
        dropout=model_config['dropout'],
        attention_dropout=model_config['attention_dropout'],
        layer_norm_eps=model_config['layer_norm_eps'],
        initializer_range=model_config['initializer_range'],
        use_cache=model_config['use_cache'],
        pad_token_id=tokenizer_config.get('pad_token_id', 3)
    )
    
    # Get parameter count for logging
    params = model.get_num_params()
    model = torch.compile(model)
    print(f"Model created: {params['total']:,} parameters")

    return model


if __name__ == "__main__":
    # Test model creation
    model = create_model()
    
    # Get model device for creating test inputs
    device = next(model.parameters()).device
    
    # Test forward pass with dummy input
    dummy_input = torch.randint(0, 16384, (2, 128), device=device)  # batch_size=2, seq_len=128
    print(f"\nTesting forward pass with input shape: {dummy_input.shape}")
    print(f"Input device: {dummy_input.device}")
    
    outputs = model(dummy_input, labels=dummy_input)
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    
    # Test generation
    print(f"\nTesting generation...")
    prompt = torch.randint(0, 16384, (1, 10), device=device)  # Single prompt of 10 tokens
    generated = model.generate(prompt, max_length=50, temperature=0.8, top_k=50)
    print(f"Generated shape: {generated.shape}")
    
    print(f"\nAll tests passed!")