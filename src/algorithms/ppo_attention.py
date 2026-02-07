"""
Modern RL algorithms with attention mechanisms.

This module implements PPO, SAC, and other state-of-the-art RL algorithms
enhanced with attention mechanisms for better state processing.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataclasses import dataclass

from .attention import MultiHeadAttention, SelfAttentionLayer, AttentionPooling


@dataclass
class PPOHyperparameters:
    """Hyperparameters for PPO algorithm."""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 64
    n_heads: int = 8
    d_model: int = 128
    dropout: float = 0.1


class AttentionPolicyNetwork(nn.Module):
    """Policy network with attention mechanism."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        hidden_dim: int = 256,
    ) -> None:
        """
        Initialize attention-based policy network.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            d_model: Model dimension for attention
            n_heads: Number of attention heads
            n_layers: Number of attention layers
            dropout: Dropout probability
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(obs_dim, d_model)
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            SelfAttentionLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Attention pooling
        self.attention_pooling = AttentionPooling(d_model, dropout)
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(
        self,
        obs: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]],
    ]:
        """
        Forward pass of policy network.
        
        Args:
            obs: Observations [batch_size, obs_dim]
            return_attention: Whether to return attention weights
            
        Returns:
            action_logits: Action logits [batch_size, action_dim]
            value: State value [batch_size, 1]
            attention_weights: List of attention weights (if return_attention=True)
        """
        batch_size = obs.size(0)
        
        # Project input to model dimension
        x = self.input_projection(obs)  # [batch_size, d_model]
        x = x.unsqueeze(1)  # [batch_size, 1, d_model] for attention
        
        attention_weights = []
        
        # Apply attention layers
        for layer in self.attention_layers:
            x, attn_weights = layer(x)
            attention_weights.append(attn_weights)
        
        # Pool attention output
        pooled, pool_weights = self.attention_pooling(x)
        attention_weights.append(pool_weights)
        
        # Get policy and value
        action_logits = self.policy_head(pooled)
        value = self.value_head(pooled)
        
        if return_attention:
            return action_logits, value, attention_weights
        else:
            return action_logits, value
    
    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            obs: Observations [batch_size, obs_dim]
            deterministic: Whether to use deterministic action selection
            
        Returns:
            action: Sampled actions [batch_size]
            log_prob: Log probabilities of actions [batch_size]
            value: State values [batch_size]
        """
        action_logits, value = self.forward(obs)
        
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=action_logits)
            action = dist.sample()
        
        log_prob = F.log_softmax(action_logits, dim=-1)
        log_prob = log_prob.gather(1, action.unsqueeze(1)).squeeze(1)
        
        return action, log_prob, value.squeeze(1)


class AttentionPPO:
    """Proximal Policy Optimization with attention mechanism."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        hyperparams: Optional[PPOHyperparameters] = None,
    ) -> None:
        """
        Initialize PPO agent with attention.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            device: Device to run on
            hyperparams: Hyperparameters
        """
        self.device = device
        self.hyperparams = hyperparams or PPOHyperparameters()
        
        # Initialize policy network
        self.policy = AttentionPolicyNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            d_model=self.hyperparams.d_model,
            n_heads=self.hyperparams.n_heads,
            dropout=self.hyperparams.dropout,
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.hyperparams.learning_rate,
        )
        
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Rewards [batch_size, seq_len]
            values: Value estimates [batch_size, seq_len]
            dones: Done flags [batch_size, seq_len]
            next_value: Next state value [batch_size]
            
        Returns:
            advantages: Computed advantages
            returns: Computed returns
        """
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute advantages using GAE
        advantage = 0
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_val = next_value
            else:
                next_val = values[:, t + 1]
            
            delta = rewards[:, t] + self.hyperparams.gamma * next_val * (1 - dones[:, t]) - values[:, t]
            advantage = delta + self.hyperparams.gamma * self.hyperparams.gae_lambda * (1 - dones[:, t]) * advantage
            advantages[:, t] = advantage
            
        # Compute returns
        returns = advantages + values
        
        return advantages, returns
    
    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Update policy using PPO.
        
        Args:
            obs: Observations [batch_size, seq_len, obs_dim]
            actions: Actions [batch_size, seq_len]
            rewards: Rewards [batch_size, seq_len]
            dones: Done flags [batch_size, seq_len]
            old_log_probs: Old log probabilities [batch_size, seq_len]
            old_values: Old value estimates [batch_size, seq_len]
            
        Returns:
            Dictionary of training metrics
        """
        batch_size, seq_len = obs.shape[:2]
        
        # Flatten for processing
        obs_flat = obs.view(-1, obs.size(-1))
        actions_flat = actions.view(-1)
        old_log_probs_flat = old_log_probs.view(-1)
        old_values_flat = old_values.view(-1)
        
        # Get current policy outputs
        action_logits, values = self.policy(obs_flat)
        current_log_probs = F.log_softmax(action_logits, dim=-1)
        current_log_probs = current_log_probs.gather(1, actions_flat.unsqueeze(1)).squeeze(1)
        
        # Compute advantages and returns
        with torch.no_grad():
            _, next_value = self.policy(obs[:, -1])
            next_value = next_value.squeeze(1)
            
        advantages, returns = self.compute_gae(
            rewards, old_values, dones, next_value
        )
        advantages_flat = advantages.view(-1)
        returns_flat = returns.view(-1)
        
        # Normalize advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
        
        # PPO update
        total_loss = 0
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for _ in range(self.hyperparams.n_epochs):
            # Create mini-batches
            indices = torch.randperm(batch_size * seq_len)
            
            for start_idx in range(0, batch_size * seq_len, self.hyperparams.batch_size):
                end_idx = min(start_idx + self.hyperparams.batch_size, batch_size * seq_len)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_obs = obs_flat[batch_indices]
                batch_actions = actions_flat[batch_indices]
                batch_old_log_probs = old_log_probs_flat[batch_indices]
                batch_advantages = advantages_flat[batch_indices]
                batch_returns = returns_flat[batch_indices]
                
                # Forward pass
                action_logits, values = self.policy(batch_obs)
                current_log_probs = F.log_softmax(action_logits, dim=-1)
                current_log_probs = current_log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                
                # Compute ratios
                ratio = torch.exp(current_log_probs - batch_old_log_probs)
                
                # Policy loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.hyperparams.clip_ratio, 1 + self.hyperparams.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(1), batch_returns)
                
                # Entropy loss
                entropy = -(F.softmax(action_logits, dim=-1) * F.log_softmax(action_logits, dim=-1)).sum(dim=-1).mean()
                entropy_loss = -self.hyperparams.entropy_coef * entropy
                
                # Total loss
                loss = policy_loss + self.hyperparams.value_loss_coef * value_loss + entropy_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.hyperparams.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())
        
        return {
            'total_loss': total_loss / (self.hyperparams.n_epochs * (batch_size * seq_len // self.hyperparams.batch_size)),
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropy_losses),
        }
    
    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from policy.
        
        Args:
            obs: Observations [batch_size, obs_dim]
            deterministic: Whether to use deterministic action selection
            
        Returns:
            action: Actions
            log_prob: Log probabilities
            value: State values
        """
        return self.policy.get_action(obs, deterministic)
