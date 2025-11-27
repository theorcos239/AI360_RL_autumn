import sys
import os

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.policies import XavierPolicy, UniformPolicy, NormalPolicy
from src.utils import run_stability_experiment, plot_stability_results
from stable_baselines3 import PPO

def main():
    env_id = "CartPole-v1"
    total_timesteps = 30000
    n_seeds = 5 # Small number for quick check
    
    configs = [
        ("MlpPolicy", "Orthogonal (Default)"),
        (XavierPolicy, "Xavier"),
        (UniformPolicy, "Uniform"),
    ]
    
    print(f"Running CartPole Experiment...")
    results = run_stability_experiment(
        PPO, configs, env_id, total_timesteps, n_seeds
    )
    
    plot_stability_results(
        results, 
        f"PPO Initialization Stability on {env_id}",
        "results/cartpole_ppo_stability.png"
    )

if __name__ == "__main__":
    main()

