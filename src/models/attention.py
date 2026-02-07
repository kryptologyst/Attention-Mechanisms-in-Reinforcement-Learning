"""
Modern attention mechanisms for reinforcement learning.

This module implements various attention mechanisms specifically designed for RL,
including multi-head attention, self-attention, and cross-attention variants.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for RL state processing."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / np.sqrt(self.d_k)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            output: Attention output [batch_size, seq_len, d_model]
            attention_weights: Attention weights [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Final linear projection
        output = self.w_o(context)
        
        return output, attention_weights


class SelfAttentionLayer(nn.Module):
    """Self-attention layer for processing sequential state information."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        layer_norm: bool = True,
    ) -> None:
        """
        Initialize self-attention layer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            layer_norm: Whether to use layer normalization
        """
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.layer_norm = nn.LayerNorm(d_model) if layer_norm else None
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of self-attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            output: Self-attention output
            attention_weights: Attention weights
        """
        # Self-attention (query, key, value are all the same)
        attn_output, attention_weights = self.attention(x, x, x, mask)
        
        # Residual connection and layer norm
        if self.layer_norm is not None:
            output = self.layer_norm(x + self.dropout(attn_output))
        else:
            output = x + self.dropout(attn_output)
            
        return output, attention_weights


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for attending to different state representations."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        layer_norm: bool = True,
    ) -> None:
        """
        Initialize cross-attention layer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            layer_norm: Whether to use layer normalization
        """
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.layer_norm = nn.LayerNorm(d_model) if layer_norm else None
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of cross-attention.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, d_model]
            key_value: Key-value tensor [batch_size, seq_len_kv, d_model]
            mask: Optional attention mask
            
        Returns:
            output: Cross-attention output
            attention_weights: Attention weights
        """
        attn_output, attention_weights = self.attention(query, key_value, key_value, mask)
        
        # Residual connection and layer norm
        if self.layer_norm is not None:
            output = self.layer_norm(query + self.dropout(attn_output))
        else:
            output = query + self.dropout(attn_output)
            
        return output, attention_weights


class AttentionPooling(nn.Module):
    """Attention-based pooling for aggregating variable-length sequences."""
    
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        """
        Initialize attention pooling.
        
        Args:
            d_model: Model dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.attention_vector = nn.Parameter(torch.randn(d_model))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of attention pooling.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            pooled: Pooled representation [batch_size, d_model]
            attention_weights: Attention weights [batch_size, seq_len]
        """
        # Compute attention scores
        scores = torch.matmul(x, self.attention_vector.unsqueeze(0).unsqueeze(-1))
        scores = scores.squeeze(-1)  # [batch_size, seq_len]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Weighted sum
        pooled = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)
        
        return pooled, attention_weights


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence data."""
    
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [seq_len, batch_size, d_model]
            
        Returns:
            Output with positional encoding added
        """
        return x + self.pe[:x.size(0), :]
