#!/usr/bin/env python3
"""
Main training script for attention-based reinforcement learning.

This script demonstrates training RL agents with attention mechanisms
on various environments with comprehensive evaluation and visualization.
"""

import argparse
import random
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
from omegaconf import OmegaConf
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import our modules
from src.algorithms.ppo_attention import AttentionPPO, PPOHyperparameters
from src.utils.trainer import AttentionRLTrainer, TrainingConfig


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str) -> torch.device:
    """Get the appropriate device for training."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_str)


def create_env(env_name: str, seed: int) -> gym.Env:
    """Create and configure environment."""
    env = gym.make(env_name)
    env.reset(seed=seed)
    return env


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train attention-based RL agent")
    parser.add_argument("--config", type=str, default="configs/default.yaml", 
                       help="Path to config file")
    parser.add_argument("--env", type=str, default="CartPole-v1",
                       help="Environment name")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda, mps)")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--eval-episodes", type=int, default=10,
                       help="Number of episodes for evaluation")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create environment
    env = create_env(args.env, args.seed)
    print(f"Environment: {args.env}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create hyperparameters
    hyperparams = PPOHyperparameters(
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        n_epochs=4,
        batch_size=64,
        n_heads=8,
        d_model=128,
        dropout=0.1,
    )
    
    # Create agent
    agent = AttentionPPO(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device=device,
        hyperparams=hyperparams,
    )
    
    # Create training config
    config = TrainingConfig(
        env_name=args.env,
        total_timesteps=args.timesteps,
        eval_freq=min(10000, args.timesteps // 10),
        n_eval_episodes=args.eval_episodes,
        save_freq=args.timesteps // 2,
        log_freq=1000,
        seed=args.seed,
        device=args.device,
        render=False,
        verbose=args.verbose,
    )
    
    # Create trainer
    trainer = AttentionRLTrainer(
        agent=agent,
        env=env,
        config=config,
        buffer_size=10000,
    )
    
    print("\nStarting training...")
    print(f"Total timesteps: {args.timesteps}")
    print(f"Evaluation frequency: {config.eval_freq}")
    print(f"Evaluation episodes: {args.eval_episodes}")
    
    # Train agent
    results = trainer.train()
    
    # Save results
    output_dir = Path(args.output_dir) / f"{args.env}_{args.seed}"
    trainer.save_results(output_dir)
    
    print(f"\nTraining completed!")
    print(f"Results saved to: {output_dir}")
    
    # Print final evaluation results
    if results['eval_metrics']:
        final_eval = results['eval_metrics'][-1]
        print(f"\nFinal evaluation results:")
        print(f"Mean reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
        print(f"Mean episode length: {final_eval['mean_length']:.2f} ± {final_eval['std_length']:.2f}")
        print(f"Min reward: {final_eval['min_reward']:.2f}")
        print(f"Max reward: {final_eval['max_reward']:.2f}")
    
    # Clean up
    env.close()


if __name__ == "__main__":
    main()
