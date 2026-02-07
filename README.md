# Attention Mechanisms in Reinforcement Learning

Research-ready implementation of reinforcement learning algorithms enhanced with attention mechanisms. This project demonstrates how attention mechanisms can improve RL agents' ability to focus on relevant parts of the state space for better decision making.

## Features

- **Modern Attention Mechanisms**: Multi-head attention, self-attention, cross-attention, and attention pooling
- **State-of-the-art RL Algorithms**: PPO with attention-enhanced policy networks
- **Comprehensive Evaluation**: Statistical analysis with confidence intervals, ablation studies
- **Interactive Demo**: Streamlit-based visualization of attention weights and training progress
- **Reproducible Research**: Deterministic seeding, structured logging, and comprehensive metrics
- **Production-ready Structure**: Clean code architecture with type hints and documentation

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Attention-Mechanisms-in-Reinforcement-Learning.git
cd Attention-Mechanisms-in-Reinforcement-Learning

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Basic Usage

```bash
# Train an attention-based PPO agent on CartPole
python scripts/train.py --env CartPole-v1 --timesteps 50000 --seed 42

# Run the interactive demo
streamlit run demo/app.py
```

### Python API

```python
from src.algorithms.ppo_attention import AttentionPPO, PPOHyperparameters
from src.utils.trainer import AttentionRLTrainer, TrainingConfig
import gymnasium as gym

# Create environment
env = gym.make("CartPole-v1")

# Create agent with attention
hyperparams = PPOHyperparameters(n_heads=8, d_model=128)
agent = AttentionPPO(
    obs_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    hyperparams=hyperparams
)

# Train the agent
config = TrainingConfig(total_timesteps=50000)
trainer = AttentionRLTrainer(agent, env, config)
results = trainer.train()
```

## Project Structure

```
attention-rl/
├── src/                          # Source code
│   ├── algorithms/               # RL algorithms
│   │   └── ppo_attention.py     # PPO with attention
│   ├── models/                   # Neural network models
│   │   └── attention.py          # Attention mechanisms
│   ├── environments/             # Custom environments
│   ├── utils/                    # Utilities
│   │   └── trainer.py           # Training utilities
│   └── evaluation/               # Evaluation metrics
├── configs/                      # Configuration files
│   └── default.yaml             # Default training config
├── scripts/                      # Training scripts
│   └── train.py                 # Main training script
├── demo/                         # Interactive demo
│   └── app.py                   # Streamlit app
├── tests/                        # Unit tests
├── notebooks/                    # Jupyter notebooks
├── assets/                       # Generated plots and results
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## Attention Mechanisms

This project implements several attention mechanisms specifically designed for RL:

### Multi-Head Attention
- Allows the model to attend to different parts of the input simultaneously
- Each head can focus on different aspects of the state space
- Configurable number of heads and model dimensions

### Self-Attention
- Enables the agent to relate different parts of the current state
- Useful for sequential decision making and state relationships
- Includes residual connections and layer normalization

### Cross-Attention
- Allows attending to different state representations
- Useful for multi-modal or hierarchical state processing
- Can be used for attention between different time steps

### Attention Pooling
- Aggregates variable-length sequences using attention weights
- Learns which parts of the sequence are most important
- Provides interpretable attention weights

## Algorithms

### Attention-Enhanced PPO
- Proximal Policy Optimization with attention-based policy networks
- Multi-head attention for state processing
- Configurable attention architecture (heads, layers, dimensions)
- Comprehensive hyperparameter tuning

## Environments

Currently supports:
- **CartPole-v1**: Classic control problem for testing basic RL
- **Acrobot-v1**: Underactuated pendulum swing-up task
- **MountainCar-v0**: Continuous control with sparse rewards

## Evaluation Metrics

### Learning Metrics
- Average return with 95% confidence intervals
- Sample efficiency (steps to reach performance threshold)
- Training stability (loss curves, entropy)
- Convergence analysis

### Attention Analysis
- Attention weight visualization
- Feature importance analysis
- Attention pattern interpretation
- Ablation studies (with/without attention)

### Robustness Testing
- Multiple random seeds
- Domain randomization
- Sensitivity analysis
- Generalization across environments

## Interactive Demo

The Streamlit demo provides:

### Training Interface
- Real-time training progress
- Configurable hyperparameters
- Multiple environment support
- Live metrics visualization

### Attention Visualization
- Interactive attention weight heatmaps
- Multi-head attention analysis
- Attention pooling visualization
- Action probability distributions

### Evaluation Dashboard
- Training curve analysis
- Statistical significance testing
- Performance comparison
- Model architecture inspection

## Configuration

Training can be configured via YAML files or command-line arguments:

```yaml
# configs/default.yaml
env_name: "CartPole-v1"
total_timesteps: 100000
eval_freq: 10000
n_eval_episodes: 10

ppo:
  learning_rate: 3.0e-4
  gamma: 0.99
  clip_ratio: 0.2

attention:
  n_heads: 8
  d_model: 128
  n_layers: 2
  dropout: 0.1
```

## Reproducibility

This project emphasizes reproducible research:

- **Deterministic Seeding**: All random number generators seeded consistently
- **Structured Logging**: Comprehensive metrics and checkpointing
- **Version Control**: Git-based experiment tracking
- **Configuration Management**: YAML-based hyperparameter management
- **Statistical Analysis**: Proper confidence intervals and significance testing

## Safety and Limitations

**Important Disclaimer**: This project is designed for research and educational purposes only. The models and algorithms implemented here are NOT intended for production use in real-world control systems, especially in safety-critical applications such as:

- Autonomous vehicles
- Medical devices
- Industrial control systems
- Financial trading systems
- Any system where failure could cause harm

### Known Limitations
- Models trained on toy environments may not generalize to real-world scenarios
- Attention mechanisms may not always improve performance
- Training can be computationally expensive
- Results may vary significantly across different random seeds

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper tests
4. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/ scripts/ demo/
ruff check src/ scripts/ demo/

# Type checking
mypy src/
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{attention_rl,
  title={Attention Mechanisms in Reinforcement Learning},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Attention-Mechanisms-in-Reinforcement-Learning}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Gymnasium for the RL environments
- PyTorch team for the deep learning framework
- Stable Baselines3 for RL algorithm inspiration
- Streamlit for the interactive demo framework

## References

1. Vaswani, A., et al. "Attention is all you need." NIPS 2017.
2. Schulman, J., et al. "Proximal policy optimization algorithms." arXiv 2017.
3. Mnih, V., et al. "Human-level control through deep reinforcement learning." Nature 2015.
4. Bahdanau, D., et al. "Neural machine translation by jointly learning to align and translate." ICLR 2015.
# Attention-Mechanisms-in-Reinforcement-Learning
