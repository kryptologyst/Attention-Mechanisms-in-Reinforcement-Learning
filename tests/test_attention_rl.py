"""
Unit tests for attention-based RL components.

This module contains comprehensive tests for all major components
including attention mechanisms, RL algorithms, and training utilities.
"""

import pytest
import torch
import numpy as np
import gymnasium as gym
from unittest.mock import Mock, patch

from src.models.attention import (
    MultiHeadAttention,
    SelfAttentionLayer,
    CrossAttentionLayer,
    AttentionPooling,
    PositionalEncoding,
)
from src.algorithms.ppo_attention import (
    AttentionPolicyNetwork,
    AttentionPPO,
    PPOHyperparameters,
)
from src.utils.trainer import (
    ReplayBuffer,
    AttentionRLTrainer,
    TrainingConfig,
)


class TestAttentionMechanisms:
    """Test attention mechanism implementations."""
    
    def test_multi_head_attention(self):
        """Test multi-head attention forward pass."""
        batch_size, seq_len, d_model = 2, 10, 64
        n_heads = 8
        
        attention = MultiHeadAttention(d_model, n_heads)
        
        # Create test inputs
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        
        # Forward pass
        output, weights = attention(query, key, value)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, d_model)
        assert weights.shape == (batch_size, n_heads, seq_len, seq_len)
        
        # Check attention weights sum to 1
        assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))
    
    def test_self_attention_layer(self):
        """Test self-attention layer."""
        batch_size, seq_len, d_model = 2, 10, 64
        n_heads = 8
        
        layer = SelfAttentionLayer(d_model, n_heads)
        
        x = torch.randn(batch_size, seq_len, d_model)
        output, weights = layer(x)
        
        assert output.shape == x.shape
        assert weights.shape == (batch_size, n_heads, seq_len, seq_len)
    
    def test_cross_attention_layer(self):
        """Test cross-attention layer."""
        batch_size, seq_len_q, seq_len_kv, d_model = 2, 5, 10, 64
        n_heads = 8
        
        layer = CrossAttentionLayer(d_model, n_heads)
        
        query = torch.randn(batch_size, seq_len_q, d_model)
        key_value = torch.randn(batch_size, seq_len_kv, d_model)
        
        output, weights = layer(query, key_value)
        
        assert output.shape == query.shape
        assert weights.shape == (batch_size, n_heads, seq_len_q, seq_len_kv)
    
    def test_attention_pooling(self):
        """Test attention pooling."""
        batch_size, seq_len, d_model = 2, 10, 64
        
        pooling = AttentionPooling(d_model)
        
        x = torch.randn(batch_size, seq_len, d_model)
        pooled, weights = pooling(x)
        
        assert pooled.shape == (batch_size, d_model)
        assert weights.shape == (batch_size, seq_len)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size))
    
    def test_positional_encoding(self):
        """Test positional encoding."""
        seq_len, d_model = 10, 64
        
        pe = PositionalEncoding(d_model)
        
        x = torch.randn(seq_len, 1, d_model)
        output = pe(x)
        
        assert output.shape == x.shape
        assert not torch.equal(output, x)  # Should be different due to encoding


class TestAttentionPolicyNetwork:
    """Test attention-based policy network."""
    
    def test_policy_network_forward(self):
        """Test policy network forward pass."""
        obs_dim, action_dim = 4, 2
        batch_size = 8
        
        policy = AttentionPolicyNetwork(obs_dim, action_dim)
        
        obs = torch.randn(batch_size, obs_dim)
        action_logits, value = policy(obs)
        
        assert action_logits.shape == (batch_size, action_dim)
        assert value.shape == (batch_size, 1)
    
    def test_policy_network_with_attention(self):
        """Test policy network with attention weights."""
        obs_dim, action_dim = 4, 2
        batch_size = 8
        
        policy = AttentionPolicyNetwork(obs_dim, action_dim)
        
        obs = torch.randn(batch_size, obs_dim)
        action_logits, value, attention_weights = policy(obs, return_attention=True)
        
        assert action_logits.shape == (batch_size, action_dim)
        assert value.shape == (batch_size, 1)
        assert isinstance(attention_weights, list)
        assert len(attention_weights) > 0
    
    def test_get_action(self):
        """Test action sampling."""
        obs_dim, action_dim = 4, 2
        batch_size = 8
        
        policy = AttentionPolicyNetwork(obs_dim, action_dim)
        
        obs = torch.randn(batch_size, obs_dim)
        action, log_prob, value = policy.get_action(obs)
        
        assert action.shape == (batch_size,)
        assert log_prob.shape == (batch_size,)
        assert value.shape == (batch_size,)
        assert torch.all(action >= 0) and torch.all(action < action_dim)


class TestAttentionPPO:
    """Test attention-based PPO algorithm."""
    
    def test_ppo_initialization(self):
        """Test PPO agent initialization."""
        obs_dim, action_dim = 4, 2
        device = torch.device("cpu")
        
        agent = AttentionPPO(obs_dim, action_dim, device)
        
        assert agent.device == device
        assert isinstance(agent.policy, AttentionPolicyNetwork)
        assert isinstance(agent.optimizer, torch.optim.Adam)
    
    def test_compute_gae(self):
        """Test GAE computation."""
        obs_dim, action_dim = 4, 2
        device = torch.device("cpu")
        
        agent = AttentionPPO(obs_dim, action_dim, device)
        
        batch_size, seq_len = 2, 10
        rewards = torch.randn(batch_size, seq_len)
        values = torch.randn(batch_size, seq_len)
        dones = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        next_value = torch.randn(batch_size)
        
        advantages, returns = agent.compute_gae(rewards, values, dones, next_value)
        
        assert advantages.shape == rewards.shape
        assert returns.shape == rewards.shape
    
    def test_get_action(self):
        """Test action selection."""
        obs_dim, action_dim = 4, 2
        device = torch.device("cpu")
        
        agent = AttentionPPO(obs_dim, action_dim, device)
        
        obs = torch.randn(1, obs_dim)
        action, log_prob, value = agent.get_action(obs)
        
        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert value.shape == (1,)


class TestReplayBuffer:
    """Test replay buffer functionality."""
    
    def test_buffer_initialization(self):
        """Test buffer initialization."""
        obs_dim, action_dim = 4, 2
        max_size = 1000
        device = torch.device("cpu")
        
        buffer = ReplayBuffer(obs_dim, action_dim, max_size, device)
        
        assert buffer.max_size == max_size
        assert buffer.device == device
        assert buffer.size == 0
        assert buffer.ptr == 0
    
    def test_add_experience(self):
        """Test adding experiences to buffer."""
        obs_dim, action_dim = 4, 2
        max_size = 1000
        device = torch.device("cpu")
        
        buffer = ReplayBuffer(obs_dim, action_dim, max_size, device)
        
        # Add experience
        obs = torch.randn(obs_dim)
        action = torch.tensor(1)
        reward = 1.0
        next_obs = torch.randn(obs_dim)
        done = False
        log_prob = torch.tensor(0.5)
        value = torch.tensor(0.8)
        
        buffer.add(obs, action, reward, next_obs, done, log_prob, value)
        
        assert buffer.size == 1
        assert buffer.ptr == 1
    
    def test_sample_batch(self):
        """Test sampling batch from buffer."""
        obs_dim, action_dim = 4, 2
        max_size = 1000
        device = torch.device("cpu")
        
        buffer = ReplayBuffer(obs_dim, action_dim, max_size, device)
        
        # Add some experiences
        for i in range(10):
            obs = torch.randn(obs_dim)
            action = torch.tensor(i % action_dim)
            reward = float(i)
            next_obs = torch.randn(obs_dim)
            done = i == 9
            log_prob = torch.tensor(0.5)
            value = torch.tensor(0.8)
            
            buffer.add(obs, action, reward, next_obs, done, log_prob, value)
        
        # Sample batch
        batch = buffer.sample(5)
        
        assert len(batch) == 7  # obs, actions, rewards, next_obs, dones, log_probs, values
        assert batch['obs'].shape == (5, obs_dim)
        assert batch['actions'].shape == (5,)


class TestTrainingConfig:
    """Test training configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        
        assert config.env_name == "CartPole-v1"
        assert config.total_timesteps == 100000
        assert config.eval_freq == 10000
        assert config.n_eval_episodes == 10
        assert config.seed == 42
        assert config.device == "auto"
        assert config.verbose == True


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_training_loop_integration(self):
        """Test complete training loop integration."""
        # Create environment
        env = gym.make("CartPole-v1")
        
        # Create agent
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        device = torch.device("cpu")
        
        agent = AttentionPPO(obs_dim, action_dim, device)
        
        # Create trainer
        config = TrainingConfig(total_timesteps=1000, eval_freq=500)
        trainer = AttentionRLTrainer(agent, env, config)
        
        # Test rollout collection
        rollout_data = trainer.collect_rollout(100)
        
        assert len(rollout_data['obs']) == 100
        assert len(rollout_data['actions']) == 100
        assert len(rollout_data['rewards']) == 100
        
        env.close()
    
    def test_evaluation_integration(self):
        """Test evaluation integration."""
        # Create environment
        env = gym.make("CartPole-v1")
        
        # Create agent
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        device = torch.device("cpu")
        
        agent = AttentionPPO(obs_dim, action_dim, device)
        
        # Create trainer
        config = TrainingConfig()
        trainer = AttentionRLTrainer(agent, env, config)
        
        # Test evaluation
        eval_metrics = trainer.evaluate(n_episodes=5)
        
        assert 'mean_reward' in eval_metrics
        assert 'std_reward' in eval_metrics
        assert 'mean_length' in eval_metrics
        assert 'std_length' in eval_metrics
        
        env.close()


if __name__ == "__main__":
    pytest.main([__file__])
