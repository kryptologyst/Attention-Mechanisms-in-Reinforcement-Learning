"""
Attention mechanisms for reinforcement learning.

This package provides modern attention mechanisms specifically designed
for RL applications, including multi-head attention, self-attention,
cross-attention, and attention pooling.
"""

from .attention import (
    MultiHeadAttention,
    SelfAttentionLayer,
    CrossAttentionLayer,
    AttentionPooling,
    PositionalEncoding,
)

__all__ = [
    "MultiHeadAttention",
    "SelfAttentionLayer", 
    "CrossAttentionLayer",
    "AttentionPooling",
    "PositionalEncoding",
]
