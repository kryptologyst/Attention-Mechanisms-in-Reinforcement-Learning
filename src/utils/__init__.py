"""
Utilities for training and evaluation.

This package provides comprehensive training utilities, evaluation metrics,
and statistical analysis tools for RL experiments with attention mechanisms.
"""

from .trainer import (
    ReplayBuffer,
    AttentionRLTrainer,
    TrainingConfig,
)

__all__ = [
    "ReplayBuffer",
    "AttentionRLTrainer", 
    "TrainingConfig",
]
