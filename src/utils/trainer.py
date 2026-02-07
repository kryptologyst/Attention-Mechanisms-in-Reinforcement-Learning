"""
Training and evaluation utilities for attention-based RL.

This module provides comprehensive training loops, evaluation metrics,
and statistical analysis for RL experiments with attention mechanisms.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import gymnasium as gym

from ..algorithms.ppo_attention import AttentionPPO, PPOHyperparameters


@dataclass
class TrainingConfig:
    """Configuration for training RL agents."""
    env_name: str = "CartPole-v1"
    total_timesteps: int = 100000
    eval_freq: int = 10000
    n_eval_episodes: int = 10
    save_freq: int = 50000
    log_freq: int = 1000
    seed: int = 42
    device: str = "auto"
    render: bool = False
    verbose: bool = True


class ReplayBuffer:
    """Experience replay buffer for collecting training data."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_size: int = 100000,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Initialize replay buffer.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            max_size: Maximum buffer size
            device: Device to store tensors on
        """
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Initialize storage
        self.obs = torch.zeros((max_size, obs_dim), device=device)
        self.actions = torch.zeros((max_size,), dtype=torch.long, device=device)
        self.rewards = torch.zeros((max_size,), device=device)
        self.next_obs = torch.zeros((max_size, obs_dim), device=device)
        self.dones = torch.zeros((max_size,), dtype=torch.bool, device=device)
        self.log_probs = torch.zeros((max_size,), device=device)
        self.values = torch.zeros((max_size,), device=device)
        
    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_obs: torch.Tensor,
        done: bool,
        log_prob: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """
        Add experience to buffer.
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode ended
            log_prob: Log probability of action
            value: State value estimate
        """
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample batch from buffer.
        
        Args:
            batch_size: Size of batch to sample
            
        Returns:
            Dictionary containing sampled experiences
        """
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return {
            'obs': self.obs[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_obs': self.next_obs[indices],
            'dones': self.dones[indices],
            'log_probs': self.log_probs[indices],
            'values': self.values[indices],
        }
    
    def get_all(self) -> Dict[str, torch.Tensor]:
        """
        Get all experiences from buffer.
        
        Returns:
            Dictionary containing all experiences
        """
        return {
            'obs': self.obs[:self.size],
            'actions': self.actions[:self.size],
            'rewards': self.rewards[:self.size],
            'next_obs': self.next_obs[:self.size],
            'dones': self.dones[:self.size],
            'log_probs': self.log_probs[:self.size],
            'values': self.values[:self.size],
        }


class AttentionRLTrainer:
    """Trainer for attention-based RL agents."""
    
    def __init__(
        self,
        agent: AttentionPPO,
        env: gym.Env,
        config: TrainingConfig,
        buffer_size: int = 10000,
    ) -> None:
        """
        Initialize trainer.
        
        Args:
            agent: RL agent to train
            env: Environment to train on
            config: Training configuration
            buffer_size: Size of replay buffer
        """
        self.agent = agent
        self.env = env
        self.config = config
        self.device = agent.device
        
        # Initialize replay buffer
        obs_dim = env.observation_space.shape[0]
        self.buffer = ReplayBuffer(
            obs_dim=obs_dim,
            action_dim=env.action_space.n,
            max_size=buffer_size,
            device=self.device,
        )
        
        # Training metrics
        self.training_metrics = []
        self.eval_metrics = []
        
    def collect_rollout(self, n_steps: int) -> Dict[str, List]:
        """
        Collect rollout data.
        
        Args:
            n_steps: Number of steps to collect
            
        Returns:
            Dictionary containing rollout data
        """
        obs_list = []
        action_list = []
        reward_list = []
        next_obs_list = []
        done_list = []
        log_prob_list = []
        value_list = []
        
        obs, _ = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        
        for _ in range(n_steps):
            # Get action from agent
            action, log_prob, value = self.agent.get_action(obs.unsqueeze(0))
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            value = value.squeeze(0)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
            
            # Store experience
            obs_list.append(obs)
            action_list.append(action)
            reward_list.append(reward)
            next_obs_list.append(next_obs)
            done_list.append(done)
            log_prob_list.append(log_prob)
            value_list.append(value)
            
            # Add to buffer
            self.buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                log_prob=log_prob,
                value=value,
            )
            
            obs = next_obs
            
            if done:
                obs, _ = self.env.reset()
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        
        return {
            'obs': obs_list,
            'actions': action_list,
            'rewards': reward_list,
            'next_obs': next_obs_list,
            'dones': done_list,
            'log_probs': log_prob_list,
            'values': value_list,
        }
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate agent performance.
        
        Args:
            n_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary containing evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            
            episode_reward = 0
            episode_length = 0
            
            while True:
                action, _, _ = self.agent.get_action(obs.unsqueeze(0), deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
                    
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
        }
    
    def train(self) -> Dict[str, List]:
        """
        Train the agent.
        
        Returns:
            Dictionary containing training metrics
        """
        total_steps = 0
        episode_count = 0
        
        pbar = tqdm(total=self.config.total_timesteps, desc="Training")
        
        while total_steps < self.config.total_timesteps:
            # Collect rollout
            rollout_data = self.collect_rollout(min(1000, self.config.total_timesteps - total_steps))
            
            # Convert to tensors
            obs = torch.stack(rollout_data['obs'])
            actions = torch.stack(rollout_data['actions'])
            rewards = torch.tensor(rollout_data['rewards'], device=self.device)
            dones = torch.tensor(rollout_data['dones'], device=self.device)
            log_probs = torch.stack(rollout_data['log_probs'])
            values = torch.stack(rollout_data['values'])
            
            # Reshape for batch processing
            batch_size = 1
            seq_len = len(rollout_data['obs'])
            
            obs = obs.unsqueeze(0)  # [1, seq_len, obs_dim]
            actions = actions.unsqueeze(0)  # [1, seq_len]
            rewards = rewards.unsqueeze(0)  # [1, seq_len]
            dones = dones.unsqueeze(0)  # [1, seq_len]
            log_probs = log_probs.unsqueeze(0)  # [1, seq_len]
            values = values.unsqueeze(0)  # [1, seq_len]
            
            # Update agent
            metrics = self.agent.update(
                obs=obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
                old_log_probs=log_probs,
                old_values=values,
            )
            
            # Update metrics
            metrics['total_steps'] = total_steps
            metrics['episode_count'] = episode_count
            self.training_metrics.append(metrics)
            
            total_steps += seq_len
            episode_count += sum(rollout_data['dones'])
            
            pbar.update(seq_len)
            
            # Evaluation
            if total_steps % self.config.eval_freq == 0:
                eval_metrics = self.evaluate(self.config.n_eval_episodes)
                eval_metrics['total_steps'] = total_steps
                self.eval_metrics.append(eval_metrics)
                
                if self.config.verbose:
                    print(f"\nEvaluation at step {total_steps}:")
                    print(f"Mean reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
                    print(f"Mean length: {eval_metrics['mean_length']:.2f} ± {eval_metrics['std_length']:.2f}")
        
        pbar.close()
        
        return {
            'training_metrics': self.training_metrics,
            'eval_metrics': self.eval_metrics,
        }
    
    def save_results(self, save_dir: Union[str, Path]) -> None:
        """
        Save training results.
        
        Args:
            save_dir: Directory to save results to
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(save_dir / "training_metrics.json", "w") as f:
            json.dump(self.training_metrics, f, indent=2)
        
        with open(save_dir / "eval_metrics.json", "w") as f:
            json.dump(self.eval_metrics, f, indent=2)
        
        # Save plots
        self.plot_training_curves(save_dir)
        
        # Save model
        torch.save(self.agent.policy.state_dict(), save_dir / "policy.pt")
    
    def plot_training_curves(self, save_dir: Union[str, Path]) -> None:
        """
        Plot training curves.
        
        Args:
            save_dir: Directory to save plots to
        """
        save_dir = Path(save_dir)
        
        # Training metrics
        if self.training_metrics:
            df_train = pd.DataFrame(self.training_metrics)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Policy loss
            axes[0, 0].plot(df_train['total_steps'], df_train['policy_loss'])
            axes[0, 0].set_title('Policy Loss')
            axes[0, 0].set_xlabel('Steps')
            axes[0, 0].set_ylabel('Loss')
            
            # Value loss
            axes[0, 1].plot(df_train['total_steps'], df_train['value_loss'])
            axes[0, 1].set_title('Value Loss')
            axes[0, 1].set_xlabel('Steps')
            axes[0, 1].set_ylabel('Loss')
            
            # Total loss
            axes[1, 0].plot(df_train['total_steps'], df_train['total_loss'])
            axes[1, 0].set_title('Total Loss')
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Loss')
            
            # Entropy
            axes[1, 1].plot(df_train['total_steps'], df_train['entropy'])
            axes[1, 1].set_title('Entropy')
            axes[1, 1].set_xlabel('Steps')
            axes[1, 1].set_ylabel('Entropy')
            
            plt.tight_layout()
            plt.savefig(save_dir / "training_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Evaluation metrics
        if self.eval_metrics:
            df_eval = pd.DataFrame(self.eval_metrics)
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Mean reward
            axes[0].errorbar(
                df_eval['total_steps'],
                df_eval['mean_reward'],
                yerr=df_eval['std_reward'],
                capsize=5
            )
            axes[0].set_title('Evaluation Reward')
            axes[0].set_xlabel('Steps')
            axes[0].set_ylabel('Mean Reward')
            
            # Mean episode length
            axes[1].errorbar(
                df_eval['total_steps'],
                df_eval['mean_length'],
                yerr=df_eval['std_length'],
                capsize=5
            )
            axes[1].set_title('Episode Length')
            axes[1].set_xlabel('Steps')
            axes[1].set_ylabel('Mean Length')
            
            plt.tight_layout()
            plt.savefig(save_dir / "evaluation_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
