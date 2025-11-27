import sys
import os
import ale_py
import shimmy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.policies import XavierPolicy, UniformCnnPolicy, NormalCnnPolicy, BinaryPolicy
from src.utils import run_stability_experiment, plot_stability_results
from stable_baselines3 import PPO

def main():
    env_id = "PongNoFrameskip-v4"
    total_timesteps = 100000
    n_seeds = 1 # Single run for Pong is enough to see start dynamics
    
    configs = [
        ("CnnPolicy", "Orthogonal (Default)"),
        (XavierPolicy, "Xavier"),
        (UniformCnnPolicy, "Uniform [-0.05, 0.05]"),
        (NormalCnnPolicy, "Normal (0, 0.05)"),
        (BinaryPolicy, "Binary {0, 1}")
    ]
    
    # Custom hyperparameters for Pong
    hyperparams = {
        "learning_rate": 2.5e-4,
        "n_steps": 128,
        "batch_size": 256,
        "ent_coef": 0.01
    }
    
    print(f"Running Pong PPO Experiment...")
    results = run_stability_experiment(
        PPO, configs, env_id, total_timesteps, n_seeds,
        is_atari=True, hyperparams=hyperparams
    )
    
    plot_stability_results(
        results, 
        f"PPO Initialization Comparison on {env_id}",
        "results/pong_ppo_comparison.png"
    )

if __name__ == "__main__":
    main()

