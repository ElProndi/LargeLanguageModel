"""
Activation functions and gated linear units for transformer models.
This module implements SwiGLU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SwiGLU(nn.Module):
    """
    SwiGLU activation function - a gated linear unit with Swish activation.
    The gating mechanism allows the model to control information flow dynamically.
    
    Formula: SwiGLU(x) = (Swish(xW_gate) ⊙ xW_up)W_down
    where Swish(x) = x * sigmoid(x)
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        bias: bool = False,
        dropout: float = 0.0
    ):
        """
        Initialize SwiGLU module.
        
        Args:
            hidden_size: Input and output dimension
            intermediate_size: Hidden dimension (default: 4 * hidden_size * 2/3)
            bias: Whether to use bias in linear layers
            dropout: Dropout probability after activation
        """
        super().__init__()
        
        # Calculate intermediate size following LLaMA's convention
        # Standard FFN uses 4x expansion, but SwiGLU needs 2/3 of that due to gating
        if intermediate_size is None:
            # Round to nearest multiple of 256 for better GPU utilization
            intermediate_size = int(2 * hidden_size * 4 / 3)
            intermediate_size = 256 * ((intermediate_size + 255) // 256)
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Fused projection for gate and up - single GEMM for better performance
        # Output size is 2 * intermediate_size (gate + up concatenated)
        self.w_fused = nn.Linear(hidden_size, 2 * intermediate_size, bias=bias)
        self.w_down = nn.Linear(intermediate_size, hidden_size, bias=bias)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU activation.
        
        Args:
            x: Input tensor of shape (..., hidden_size)
            
        Returns:
            Output tensor of shape (..., hidden_size)
        """
        # Apply fused projection and split into gate and up components
        # Single GEMM operation
        fused_output = self.w_fused(x)
        gate, up = fused_output.chunk(2, dim=-1)
        
        # Apply Swish activation to gate using optimized F.silu
        # F.silu is PyTorch's optimized implementation of x * sigmoid(x)
        gate = F.silu(gate)
        
        # Element-wise multiplication (gating mechanism)
        # In-place operation to save memory
        gate.mul_(up)
        intermediate = gate
        
        # Apply down projection
        output = self.w_down(intermediate)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output