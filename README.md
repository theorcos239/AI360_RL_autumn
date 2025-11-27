# AI360 RL Coursework

This repo contains my experiments for the Autumn 2024 RL course.
There are two main parts here: playing with PPO/TRPO initializations and analyzing KL divergence behavior.

## Structure

### 1. Initialization Experiments (`initialization_experiments/`)
I wanted to check how much weight initialization actually matters for PPO and TRPO.
Testing Orthogonal (default in SB3) vs Xavier vs Random vs "Binary" (just to break things).

*   **Environments:** CartPole, Acrobot, and Atari Pong.
*   **Code:** Custom policy classes in `src/policies.py`.
*   **How to run:**
    ```bash
    python initialization_experiments/experiments/run_acrobot_ppo.py
    ```

### 2. KL Divergence Analysis (`kl_experiments/`)
Visualizing how KL divergence behaves for different distributions.
*(Scripts `kl_variants.py` and `plot_kl_lines.py` act as sandboxes for math tests).*

## Setup

Just install the requirements. You'll need `swig` for Box2D (LunarLander/Acrobot) if you're on Mac.

```bash
pip install -r requirements.txt
```

If `box2d-py` fails to build on Mac M1/M2, install swig via conda: `conda install swig`.

### 3. Self-Made PPO Implementation
I implemented PPO from scratch to deeply understand the algorithm. The implementation includes:
- Generalized Advantage Estimation (GAE)
- Clipped objective function
- Multiple epochs of minibatch updates
- Proper value function training


* **Results:** Pre-trained model on Pong-v5 achieving average reward of -5 (beats random policy which scores -21)
* **Architecture:** CNN-based Actor-Critic with orthogonal initialization

* **How to run training:**
  run main.ipynb in selfmade_ppo directory
