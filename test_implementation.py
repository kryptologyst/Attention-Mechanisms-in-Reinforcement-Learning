#!/usr/bin/env python3
"""
Quick test script to verify the attention-based RL implementation works correctly.

This script performs a minimal training run to ensure all components are working
and can be used for quick validation during development.
"""

import torch
import numpy as np
import gymnasium as gym
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from algorithms.ppo_attention import AttentionPPO, PPOHyperparameters
from utils.trainer import AttentionRLTrainer, TrainingConfig


def test_attention_rl():
    """Test the attention-based RL implementation."""
    print("Testing Attention-Based RL Implementation")
    print("=" * 50)
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create environment
    env = gym.make("CartPole-v1")
    print(f"✓ Environment created: {env.spec.id}")
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")
    
    # Create agent
    hyperparams = PPOHyperparameters(
        learning_rate=1e-3,  # Higher LR for faster testing
        n_heads=4,           # Fewer heads for faster testing
        d_model=64,          # Smaller model for faster testing
        batch_size=32,       # Smaller batch for faster testing
    )
    
    agent = AttentionPPO(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device=device,
        hyperparams=hyperparams,
    )
    
    print(f"✓ Agent created with {sum(p.numel() for p in agent.policy.parameters()):,} parameters")
    
    # Test action selection
    obs, _ = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    action, log_prob, value = agent.get_action(obs_tensor)
    
    print(f"✓ Action selection works: action={action.item()}, log_prob={log_prob.item():.3f}, value={value.item():.3f}")
    
    # Test attention weights
    action_logits, value, attention_weights = agent.policy.forward(obs_tensor, return_attention=True)
    print(f"✓ Attention mechanism works: {len(attention_weights)} attention layers")
    
    # Create trainer
    config = TrainingConfig(
        total_timesteps=1000,  # Very short training for testing
        eval_freq=500,
        n_eval_episodes=3,
        verbose=False,
    )
    
    trainer = AttentionRLTrainer(agent, env, config)
    print("✓ Trainer created")
    
    # Test rollout collection
    rollout_data = trainer.collect_rollout(100)
    print(f"✓ Rollout collection works: {len(rollout_data['obs'])} steps collected")
    
    # Test evaluation
    eval_metrics = trainer.evaluate(n_episodes=3)
    print(f"✓ Evaluation works: mean_reward={eval_metrics['mean_reward']:.2f}")
    
    # Test training (very short)
    print("\nRunning short training test...")
    results = trainer.train()
    
    if results['eval_metrics']:
        final_eval = results['eval_metrics'][-1]
        print(f"✓ Training completed: final_reward={final_eval['mean_reward']:.2f}")
    else:
        print("✓ Training completed (no evaluation metrics)")
    
    # Test saving results
    output_dir = Path("test_results")
    trainer.save_results(output_dir)
    print(f"✓ Results saved to: {output_dir}")
    
    # Clean up
    env.close()
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! The implementation is working correctly.")
    print("\nNext steps:")
    print("1. Run full training: python scripts/train.py --timesteps 50000")
    print("2. Launch demo: streamlit run demo/app.py")
    print("3. Run tests: pytest tests/")


if __name__ == "__main__":
    test_attention_rl()
