"""
Reinforcement learning algorithms with attention mechanisms.

This package provides state-of-the-art RL algorithms enhanced with
attention mechanisms for improved state processing and decision making.
"""

from .ppo_attention import (
    AttentionPolicyNetwork,
    AttentionPPO,
    PPOHyperparameters,
)

__all__ = [
    "AttentionPolicyNetwork",
    "AttentionPPO",
    "PPOHyperparameters",
]
