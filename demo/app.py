"""
Streamlit demo for attention-based reinforcement learning.

This interactive demo allows users to:
- Train RL agents with attention mechanisms
- Visualize attention weights and training progress
- Compare different attention architectures
- Evaluate trained models
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gymnasium as gym
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple

# Import our modules
from src.algorithms.ppo_attention import AttentionPPO, PPOHyperparameters
from src.utils.trainer import AttentionRLTrainer, TrainingConfig
from src.models.attention import MultiHeadAttention, SelfAttentionLayer


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the appropriate device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def create_env(env_name: str, seed: int) -> gym.Env:
    """Create and configure environment."""
    env = gym.make(env_name)
    env.reset(seed=seed)
    return env


def plot_attention_weights(attention_weights: torch.Tensor, title: str = "Attention Weights") -> go.Figure:
    """Plot attention weights as a heatmap."""
    weights = attention_weights.detach().cpu().numpy()
    
    fig = go.Figure(data=go.Heatmap(
        z=weights,
        colorscale='Viridis',
        showscale=True,
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Key Position",
        yaxis_title="Query Position",
        width=500,
        height=400,
    )
    
    return fig


def plot_training_metrics(metrics: List[Dict]) -> go.Figure:
    """Plot training metrics."""
    df = pd.DataFrame(metrics)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Policy Loss', 'Value Loss', 'Total Loss', 'Entropy'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Policy loss
    fig.add_trace(
        go.Scatter(x=df['total_steps'], y=df['policy_loss'], name='Policy Loss'),
        row=1, col=1
    )
    
    # Value loss
    fig.add_trace(
        go.Scatter(x=df['total_steps'], y=df['value_loss'], name='Value Loss'),
        row=1, col=2
    )
    
    # Total loss
    fig.add_trace(
        go.Scatter(x=df['total_steps'], y=df['total_loss'], name='Total Loss'),
        row=2, col=1
    )
    
    # Entropy
    fig.add_trace(
        go.Scatter(x=df['total_steps'], y=df['entropy'], name='Entropy'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    fig.update_xaxes(title_text="Steps")
    fig.update_yaxes(title_text="Loss/Entropy")
    
    return fig


def plot_evaluation_metrics(metrics: List[Dict]) -> go.Figure:
    """Plot evaluation metrics."""
    df = pd.DataFrame(metrics)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Mean Reward', 'Episode Length'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Mean reward with error bars
    fig.add_trace(
        go.Scatter(
            x=df['total_steps'],
            y=df['mean_reward'],
            error_y=dict(type='data', array=df['std_reward']),
            name='Mean Reward',
            mode='lines+markers'
        ),
        row=1, col=1
    )
    
    # Episode length with error bars
    fig.add_trace(
        go.Scatter(
            x=df['total_steps'],
            y=df['mean_length'],
            error_y=dict(type='data', array=df['std_length']),
            name='Episode Length',
            mode='lines+markers'
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(title_text="Steps")
    fig.update_yaxes(title_text="Reward/Length")
    
    return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Attention-Based RL Demo",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Attention Mechanisms in Reinforcement Learning")
    st.markdown("""
    This demo showcases reinforcement learning agents enhanced with attention mechanisms.
    Attention allows agents to focus on the most relevant parts of the state space for decision making.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Environment selection
    env_name = st.sidebar.selectbox(
        "Environment",
        ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"],
        index=0
    )
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    total_timesteps = st.sidebar.slider("Total Timesteps", 10000, 200000, 50000)
    eval_freq = st.sidebar.slider("Evaluation Frequency", 1000, 20000, 5000)
    n_eval_episodes = st.sidebar.slider("Evaluation Episodes", 5, 20, 10)
    seed = st.sidebar.number_input("Random Seed", 0, 1000, 42)
    
    # Attention parameters
    st.sidebar.subheader("Attention Parameters")
    n_heads = st.sidebar.slider("Number of Heads", 1, 16, 8)
    d_model = st.sidebar.slider("Model Dimension", 64, 512, 128)
    n_layers = st.sidebar.slider("Number of Layers", 1, 6, 2)
    dropout = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.1)
    
    # PPO parameters
    st.sidebar.subheader("PPO Parameters")
    learning_rate = st.sidebar.slider("Learning Rate", 1e-5, 1e-2, 3e-4, format="%.2e")
    gamma = st.sidebar.slider("Discount Factor", 0.9, 0.999, 0.99)
    clip_ratio = st.sidebar.slider("Clip Ratio", 0.1, 0.3, 0.2)
    entropy_coef = st.sidebar.slider("Entropy Coefficient", 0.0, 0.1, 0.01)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Training", "Attention Visualization", "Evaluation", "Model Analysis"])
    
    with tab1:
        st.header("🚀 Training")
        
        if st.button("Start Training", type="primary"):
            # Set seed
            set_seed(seed)
            
            # Get device
            device = get_device()
            st.info(f"Using device: {device}")
            
            # Create environment
            env = create_env(env_name, seed)
            st.info(f"Environment: {env_name}")
            st.info(f"Observation space: {env.observation_space.shape}")
            st.info(f"Action space: {env.action_space.n}")
            
            # Create hyperparameters
            hyperparams = PPOHyperparameters(
                learning_rate=learning_rate,
                gamma=gamma,
                clip_ratio=clip_ratio,
                entropy_coef=entropy_coef,
                n_heads=n_heads,
                d_model=d_model,
                dropout=dropout,
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
                env_name=env_name,
                total_timesteps=total_timesteps,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                seed=seed,
                device=str(device),
                verbose=False,
            )
            
            # Create trainer
            trainer = AttentionRLTrainer(
                agent=agent,
                env=env,
                config=config,
                buffer_size=10000,
            )
            
            # Training progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Train agent
            results = trainer.train()
            
            # Store results in session state
            st.session_state['training_results'] = results
            st.session_state['agent'] = agent
            st.session_state['env'] = env
            
            progress_bar.progress(1.0)
            status_text.text("Training completed!")
            
            # Display final results
            if results['eval_metrics']:
                final_eval = results['eval_metrics'][-1]
                st.success(f"Final evaluation results:")
                st.success(f"Mean reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
                st.success(f"Mean episode length: {final_eval['mean_length']:.2f} ± {final_eval['std_length']:.2f}")
            
            env.close()
    
    with tab2:
        st.header("🎯 Attention Visualization")
        
        if 'agent' in st.session_state and 'env' in st.session_state:
            agent = st.session_state['agent']
            env = st.session_state['env']
            
            # Generate attention weights
            obs, _ = env.reset()
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
            
            # Get attention weights
            action_logits, value, attention_weights = agent.policy.forward(obs_tensor, return_attention=True)
            
            # Plot attention weights
            for i, weights in enumerate(attention_weights[:-1]):  # Exclude pooling weights
                if weights.dim() == 4:  # Multi-head attention weights
                    # Average across heads
                    avg_weights = weights.mean(dim=1).squeeze(0)
                    fig = plot_attention_weights(avg_weights, f"Attention Layer {i+1}")
                    st.plotly_chart(fig, use_container_width=True)
                elif weights.dim() == 2:  # Pooling weights
                    fig = go.Figure(data=go.Bar(y=weights.squeeze().detach().cpu().numpy()))
                    fig.update_layout(
                        title=f"Attention Pooling Weights",
                        xaxis_title="Feature Index",
                        yaxis_title="Attention Weight",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Show action probabilities
            action_probs = torch.softmax(action_logits, dim=-1).squeeze().detach().cpu().numpy()
            fig = go.Figure(data=go.Bar(x=list(range(len(action_probs))), y=action_probs))
            fig.update_layout(
                title="Action Probabilities",
                xaxis_title="Action",
                yaxis_title="Probability",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("Please train a model first to visualize attention weights.")
    
    with tab3:
        st.header("📊 Evaluation")
        
        if 'training_results' in st.session_state:
            results = st.session_state['training_results']
            
            # Training metrics
            if results['training_metrics']:
                st.subheader("Training Metrics")
                fig = plot_training_metrics(results['training_metrics'])
                st.plotly_chart(fig, use_container_width=True)
            
            # Evaluation metrics
            if results['eval_metrics']:
                st.subheader("Evaluation Metrics")
                fig = plot_evaluation_metrics(results['eval_metrics'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                df_eval = pd.DataFrame(results['eval_metrics'])
                st.subheader("Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Best Mean Reward", f"{df_eval['mean_reward'].max():.2f}")
                with col2:
                    st.metric("Final Mean Reward", f"{df_eval['mean_reward'].iloc[-1]:.2f}")
                with col3:
                    st.metric("Best Episode Length", f"{df_eval['mean_length'].max():.2f}")
                with col4:
                    st.metric("Final Episode Length", f"{df_eval['mean_length'].iloc[-1]:.2f}")
        else:
            st.info("Please train a model first to view evaluation results.")
    
    with tab4:
        st.header("🔍 Model Analysis")
        
        if 'agent' in st.session_state:
            agent = st.session_state['agent']
            
            # Model architecture
            st.subheader("Model Architecture")
            
            # Count parameters
            total_params = sum(p.numel() for p in agent.policy.parameters())
            trainable_params = sum(p.numel() for p in agent.policy.parameters() if p.requires_grad)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Parameters", f"{total_params:,}")
            with col2:
                st.metric("Trainable Parameters", f"{trainable_params:,}")
            
            # Model structure
            st.subheader("Model Structure")
            model_str = str(agent.policy)
            st.code(model_str, language="python")
            
            # Hyperparameters
            st.subheader("Hyperparameters")
            hyperparams_dict = {
                "Learning Rate": agent.hyperparams.learning_rate,
                "Gamma": agent.hyperparams.gamma,
                "GAE Lambda": agent.hyperparams.gae_lambda,
                "Clip Ratio": agent.hyperparams.clip_ratio,
                "Value Loss Coef": agent.hyperparams.value_loss_coef,
                "Entropy Coef": agent.hyperparams.entropy_coef,
                "Max Grad Norm": agent.hyperparams.max_grad_norm,
                "N Epochs": agent.hyperparams.n_epochs,
                "Batch Size": agent.hyperparams.batch_size,
                "N Heads": agent.hyperparams.n_heads,
                "D Model": agent.hyperparams.d_model,
                "Dropout": agent.hyperparams.dropout,
            }
            
            df_hyperparams = pd.DataFrame(list(hyperparams_dict.items()), columns=["Parameter", "Value"])
            st.dataframe(df_hyperparams, use_container_width=True)
            
        else:
            st.info("Please train a model first to view model analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Disclaimer**: This is a research/educational demonstration. 
    The models trained here are not intended for production use in real-world control systems.
    """)


if __name__ == "__main__":
    main()
