"""
Modern activation functions and gated linear units for transformer models.

This module implements SwiGLU and other gated activation functions that have
shown superior performance in recent large language models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SwiGLU(nn.Module):
    """
    SwiGLU activation function - a gated linear unit with Swish activation.
    
    Used in models like LLaMA, PaLM, and others for improved performance.
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
        
        # Three projection layers for SwiGLU
        self.w_gate = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.w_up = nn.Linear(hidden_size, intermediate_size, bias=bias)
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
        # Apply gate and up projections
        gate = self.w_gate(x)
        up = self.w_up(x)
        
        # Apply Swish activation to gate and element-wise multiply with up
        # Swish(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
        gate = F.silu(gate)  # SiLU is equivalent to Swish
        
        # Element-wise multiplication (gating mechanism)
        intermediate = gate * up
        
        # Apply down projection
        output = self.w_down(intermediate)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output
    
    def get_num_params(self) -> dict:
        """Get parameter count breakdown."""
        gate_params = self.w_gate.weight.numel()
        if self.w_gate.bias is not None:
            gate_params += self.w_gate.bias.numel()
        
        up_params = self.w_up.weight.numel()
        if self.w_up.bias is not None:
            up_params += self.w_up.bias.numel()
        
        down_params = self.w_down.weight.numel()
        if self.w_down.bias is not None:
            down_params += self.w_down.bias.numel()
        
        return {
            'gate': gate_params,
            'up': up_params,
            'down': down_params,
            'total': gate_params + up_params + down_params
        }


class GeGLU(nn.Module):
    """
    GeGLU activation function - a gated linear unit with GELU activation.
    
    Similar to SwiGLU but uses GELU instead of Swish for the gating activation.
    Formula: GeGLU(x) = (GELU(xW_gate) ⊙ xW_up)W_down
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        bias: bool = False,
        dropout: float = 0.0
    ):
        """
        Initialize GeGLU module.
        
        Args:
            hidden_size: Input and output dimension
            intermediate_size: Hidden dimension
            bias: Whether to use bias in linear layers
            dropout: Dropout probability after activation
        """
        super().__init__()
        
        if intermediate_size is None:
            intermediate_size = int(2 * hidden_size * 4 / 3)
            intermediate_size = 256 * ((intermediate_size + 255) // 256)
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        self.w_gate = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.w_up = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.w_down = nn.Linear(intermediate_size, hidden_size, bias=bias)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GeGLU activation."""
        gate = F.gelu(self.w_gate(x))
        up = self.w_up(x)
        intermediate = gate * up
        output = self.w_down(intermediate)
        return self.dropout(output)


class ReGLU(nn.Module):
    """
    ReGLU activation function - a gated linear unit with ReLU activation.
    
    Formula: ReGLU(x) = (ReLU(xW_gate) ⊙ xW_up)W_down
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        bias: bool = False,
        dropout: float = 0.0
    ):
        """Initialize ReGLU module."""
        super().__init__()
        
        if intermediate_size is None:
            intermediate_size = int(2 * hidden_size * 4 / 3)
            intermediate_size = 256 * ((intermediate_size + 255) // 256)
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        self.w_gate = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.w_up = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.w_down = nn.Linear(intermediate_size, hidden_size, bias=bias)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ReGLU activation."""
        gate = F.relu(self.w_gate(x))
        up = self.w_up(x)
        intermediate = gate * up
        output = self.w_down(intermediate)
        return self.dropout(output)


def test_swiglu():
    """Test function to verify SwiGLU implementation."""
    print("Testing SwiGLU implementation...")
    
    # Test dimensions
    batch_size = 2
    seq_len = 128
    hidden_size = 512
    
    # Create dummy input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # Initialize SwiGLU module
    swiglu = SwiGLU(
        hidden_size=hidden_size,
        intermediate_size=None,  # Will auto-calculate
        bias=False,
        dropout=0.1
    ).to(device)
    
    # Forward pass
    output = swiglu(x)
    
    # Verify shape preservation
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
    
    # Verify gradient flow
    output.sum().backward()
    assert swiglu.w_gate.weight.grad is not None, "No gradients for gate weights"
    assert swiglu.w_up.weight.grad is not None, "No gradients for up weights"
    assert swiglu.w_down.weight.grad is not None, "No gradients for down weights"
    
    # Get parameter count
    params = swiglu.get_num_params()
    
    print("✓ All SwiGLU tests passed!")
    print(f"  - Input shape: ({batch_size}, {seq_len}, {hidden_size})")
    print(f"  - Intermediate size: {swiglu.intermediate_size}")
    print(f"  - Parameter count:")
    print(f"    - Gate projection: {params['gate']:,}")
    print(f"    - Up projection: {params['up']:,}")
    print(f"    - Down projection: {params['down']:,}")
    print(f"    - Total: {params['total']:,}")
    print(f"  - Output shape preserved: ✓")
    print(f"  - Gradient flow: ✓")
    
    # Compare with standard FFN
    standard_ffn_params = hidden_size * (hidden_size * 4) + (hidden_size * 4) * hidden_size
    print(f"\n  Comparison with standard FFN:")
    print(f"    - Standard FFN params: {standard_ffn_params:,}")
    print(f"    - SwiGLU params: {params['total']:,}")
    print(f"    - Ratio: {params['total'] / standard_ffn_params:.2f}x")


if __name__ == "__main__":
    test_swiglu()