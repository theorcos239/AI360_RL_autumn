import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from stable_baselines3 import PPO
from sb3_contrib import TRPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from typing import Dict, Tuple, List, Any, Type, Union

from .callbacks import RewardCallback, TqdmCallback

def run_single_experiment(
    algo_class: Type,
    policy_class: Any,
    env_id: str,
    total_timesteps: int,
    seed: int = 0,
    check_freq: int = 2000,
    n_eval_episodes: int = 5,
    is_atari: bool = False,
    hyperparams: Dict = None
) -> Tuple[List[int], List[float]]:
    
    if hyperparams is None:
        hyperparams = {}
        
    # Environment Setup
    if is_atari:
        # Standard Atari wrappers
        env = make_atari_env(env_id, n_envs=4, seed=seed)
        env = VecFrameStack(env, n_stack=4)
    else:
        env = gym.make(env_id)
        
    # Model Setup
    model = algo_class(policy_class, env, verbose=0, seed=seed, **hyperparams)
    
    # Training with Progress Bar
    with tqdm(total=total_timesteps, desc=f"Training", unit="steps", leave=False) as pbar:
        callback = TqdmCallback(check_freq=check_freq, pbar=pbar, n_eval_episodes=n_eval_episodes)
        model.learn(total_timesteps=total_timesteps, callback=callback)
        
    if is_atari:
        env.close()
        
    # Ensure consistent length (sometimes callback misses last step if total_timesteps is small)
    return callback.timesteps, callback.rewards

def run_stability_experiment(
    algo_class: Type,
    configs: List[Tuple[Any, str]],
    env_id: str,
    total_timesteps: int,
    n_seeds: int = 5,
    is_atari: bool = False,
    hyperparams: Dict = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    
    results = {}
    
    for policy_cls, name in configs:
        print(f"\n=== Testing {name} ({n_seeds} seeds) ===")
        all_rewards = []
        common_timesteps = None
        
        for seed in range(n_seeds):
            print(f"  Seed {seed+1}/{n_seeds}...", end="\r")
            ts, rewards = run_single_experiment(
                algo_class, policy_cls, env_id, total_timesteps, seed, 
                is_atari=is_atari, hyperparams=hyperparams
            )
            
            if common_timesteps is None:
                common_timesteps = np.array(ts)
            
            all_rewards.append(rewards)
        print("  Done.                 ")
        
        # Align lengths
        min_len = min(len(r) for r in all_rewards)
        all_rewards = [r[:min_len] for r in all_rewards]
        common_timesteps = common_timesteps[:min_len]
        
        results[name] = (common_timesteps, np.array(all_rewards))
        
    return results

def plot_stability_results(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]], 
    title: str, 
    output_file: str
):
    plt.figure(figsize=(12, 8))
    
    for name, (ts, rewards_matrix) in results.items():
        median = np.median(rewards_matrix, axis=0)
        p25 = np.percentile(rewards_matrix, 25, axis=0)
        p75 = np.percentile(rewards_matrix, 75, axis=0)
        
        plt.plot(ts, median, label=f"{name} (Median)", linewidth=2)
        plt.fill_between(ts, p25, p75, alpha=0.2)
        
    plt.title(title)
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")

