import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.policies import XavierPolicy, UniformPolicy, NormalPolicy, BinaryPolicy
from src.utils import run_stability_experiment, plot_stability_results
from sb3_contrib import TRPO

def main():
    env_id = "Acrobot-v1"
    total_timesteps = 40000
    n_seeds = 11
    
    configs = [
        ("MlpPolicy", "Orthogonal (Default)"),
        (XavierPolicy, "Xavier"),
        (UniformPolicy, "Uniform [-0.1, 0.1]"),
        (NormalPolicy, "Normal (0, 0.1)"),
        (BinaryPolicy, "Binary {0, 1}")
    ]
    
    print(f"Running Acrobot TRPO Experiment...")
    results = run_stability_experiment(
        TRPO, configs, env_id, total_timesteps, n_seeds
    )
    
    plot_stability_results(
        results, 
        f"TRPO Initialization Stability on {env_id}",
        "results/acrobot_trpo_stability.png"
    )

if __name__ == "__main__":
    main()

