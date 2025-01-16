"""
Lightning Attention implementation based on the architecture diagram.

This module implements a modified attention mechanism that includes:
- RMSNorm for input and output normalization
- SiLU activations for Q,K,V projections
- Gating mechanism with sigmoid activation
- Scaled dot-product attention with multiple heads
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import math
from loguru import logger

class RMSNorm(nn.Module):
    """Root Mean Square Normalization module.
    
    Args:
        dim (int): The dimension to normalize over
        eps (float, optional): Small value for numerical stability. Defaults to 1e-6.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        logger.debug(f"Initialized RMSNorm with dim={dim}, eps={eps}")

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization.
        
        Args:
            x (Tensor): Input tensor of shape [..., dim]
            
        Returns:
            Tensor: Normalized tensor of the same shape
        """
        # Calculate RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SiLU(nn.Module):
    """Sigmoid Linear Unit activation function.
    
    Computes x * sigmoid(x)
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply SiLU activation.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Activated tensor
        """
        return x * torch.sigmoid(x)


class LightningAttention(nn.Module):
    """Lightning Attention module with gating mechanism.
    
    This implements a modified attention mechanism with:
    - RMSNorm for input and output
    - SiLU activations for Q,K,V projections
    - Additional gating mechanism
    - Multi-head scaled dot-product attention
    
    Args:
        dim (int): Model dimension
        num_heads (int, optional): Number of attention heads. Defaults to 8.
    """
    
    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
            
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Input projections for Q, K, V
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        
        # Gating mechanism
        self.g_proj = nn.Linear(dim, dim, bias=False)
        
        # Input and output normalizations
        self.input_norm = RMSNorm(dim)
        self.final_norm = RMSNorm(dim)
        
        # Activation functions
        self.silu = SiLU()
        
        logger.info(f"Initialized LightningAttention with dim={dim}, num_heads={num_heads}")

    def _split_heads(self, x: Tensor, batch_size: int) -> Tensor:
        """Split tensor into attention heads.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, dim]
            batch_size (int): Batch size
            
        Returns:
            Tensor: Reshaped tensor of shape [batch_size, seq_len, num_heads, head_dim]
        """
        return x.view(batch_size, -1, self.num_heads, self.head_dim)

    def _compute_attention(
        self, 
        q: Tensor, 
        k: Tensor, 
        v: Tensor,
        scale: float
    ) -> Tensor:
        """Compute scaled dot-product attention.
        
        Args:
            q (Tensor): Query tensor
            k (Tensor): Key tensor
            v (Tensor): Value tensor
            scale (float): Scaling factor for dot product
            
        Returns:
            Tensor: Output of attention computation
        """
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        logger.debug(f"Attention scores shape: {attn.shape}")
        return torch.matmul(attn, v)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of Lightning Attention.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, dim]
            
        Returns:
            Tensor: Output tensor of shape [batch_size, seq_len, dim]
        """
        # Apply input normalization
        x = self.input_norm(x)
        
        B, L, D = x.shape
        logger.debug(f"Input shape: [batch_size={B}, seq_len={L}, dim={D}]")
        
        # Project queries, keys, values with SiLU activation
        q = self.silu(self.q_proj(x))
        k = self.silu(self.k_proj(x))
        v = self.silu(self.v_proj(x))
        
        # Split heads
        q = self._split_heads(q, B)
        k = self._split_heads(k, B)
        v = self._split_heads(v, B)
        
        # Calculate gating function
        g = self.g_proj(x)
        
        # Compute attention
        scale = 1.0 / math.sqrt(self.head_dim)
        out = self._compute_attention(q, k, v, scale)
        
        # Merge heads
        out = out.reshape(B, L, D)
        
        # Apply gating and sigmoid
        out = out * torch.sigmoid(g)
        
        # Final normalization
        return self.final_norm(out)

    def extra_repr(self) -> str:
        """Extra string representation of the module."""
        return f'dim={self.head_dim * self.num_heads}, num_heads={self.num_heads}'


def setup_logger() -> None:
    """Configure loguru logger."""
    logger.remove()  # Remove default handler
    logger.add(
        "lightning_attention.log",
        rotation="500 MB",
        level="DEBUG",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>"
    )


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logger()
    logger.info("Starting Lightning Attention example")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create a sample input
    batch_size = 2
    seq_length = 16
    dim = 512
    num_heads = 8
    
    try:
        # Initialize the Lightning Attention module
        lightning_attn = LightningAttention(dim=dim, num_heads=num_heads)
        logger.info("Successfully initialized Lightning Attention module")
        
        # Create random input tensor [batch_size, seq_length, dim]
        x = torch.randn(batch_size, seq_length, dim)
        logger.debug(f"Created input tensor with shape: {x.shape}")
        
        # Forward pass
        output = lightning_attn(x)
        logger.info("Successfully completed forward pass")
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        
        # Verify that the output dimensions match the input
        assert x.shape == output.shape, "Output shape should match input shape"
        logger.debug("Shape verification passed")
        
        # Example of model statistics
        num_params = sum(p.numel() for p in lightning_attn.parameters())
        logger.info(f"Model has {num_params:,} parameters")
        
        print("\nModel Statistics:")
        print(f"Number of parameters: {num_params:,}")
        print(f"Number of attention heads: {num_heads}")
        print(f"Hidden dimension per head: {dim // num_heads}")

        # Optional: show some sample attention values
        with torch.no_grad():
            B, L, D = x.shape
            H = num_heads
            
            # Get attention scores (for visualization)
            q = lightning_attn.silu(lightning_attn.q_proj(x)).view(B, L, H, -1)
            k = lightning_attn.silu(lightning_attn.k_proj(x)).view(B, L, H, -1)
            scale = 1.0 / math.sqrt(lightning_attn.head_dim)
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            
            print("\nSample attention scores (first head, first batch):")
            print(attn[0, 0, 0, :8].tolist())
            logger.debug("Successfully computed sample attention scores")
            
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        raise