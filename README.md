<<<<<<< HEAD
# AI360_RL_autumn
=======
# RL Initialization Study

This project explores the impact of **weight initialization** on the performance and stability of Reinforcement Learning algorithms (PPO and TRPO) using Stable Baselines 3.

We test different initialization strategies across environments of varying complexity: **CartPole** (Simple), **Acrobot** (Intermediate), and **Atari Pong** (Complex, CNN-based).

## Installation

1. Clone the repository.
2. Create a conda environment (optional but recommended):
   ```bash
   conda create -n rl_init python=3.10
   conda activate rl_init
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: On Apple Silicon (M1/M2), ensure `swig` is installed (`conda install swig`) for Box2D support.*

## Project Structure

```
.
├── src/                # Core logic
│   ├── policies.py     # Custom Policy classes (Xavier, Uniform, etc.)
│   ├── callbacks.py    # Training callbacks (logging, tqdm)
│   └── utils.py        # Experiment runners and plotting
├── experiments/        # Executable scripts
│   ├── run_cartpole_comparison.py
│   ├── run_acrobot_ppo.py
│   ├── run_acrobot_trpo.py
│   └── run_pong_ppo.py
└── results/            # Output plots
```

## Initializations Tested

1.  **Orthogonal (Default in SB3):** Preserves gradient norm, ideal for deep networks. Uses gain=0.01 for action layer to maximize initial exploration.
2.  **Xavier (Glorot):** Standard for Tanh activations. Balances variance across layers.
3.  **Uniform:** Random weights from `[-0.1, 0.1]` (or `[-0.05, 0.05]` for CNN). Naive approach.
4.  **Normal:** Random weights from `N(0, 0.1)`.
5.  **Binary:** Weights are randomly `{0, 1}`. Extreme case to demonstrate saturation issues.

## Usage

Run any experiment from the root directory:

```bash
# Quick test on CartPole
python experiments/run_cartpole_comparison.py

# Stability analysis on Acrobot (11 seeds)
python experiments/run_acrobot_ppo.py

# Deep RL test on Pong (requires ~20-30 min)
python experiments/run_pong_ppo.py
```

Results (plots) will be saved in the `results/` folder.

>>>>>>> 129561d (Add RL initialization experiments (PPO/TRPO stability))
