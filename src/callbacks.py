from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

class RewardCallback(BaseCallback):
    """
    Callback for recording evaluation reward during training.
    """
    def __init__(self, check_freq: int, n_eval_episodes: int = 5, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.n_eval_episodes = n_eval_episodes
        self.rewards = []
        self.timesteps = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Evaluate the agent
            # We use the training env for speed. For rigorous testing, use a separate eval env.
            mean_reward, _ = evaluate_policy(self.model, self.training_env, n_eval_episodes=self.n_eval_episodes, warn=False)
            self.rewards.append(mean_reward)
            self.timesteps.append(self.num_timesteps)
        return True

class TqdmCallback(RewardCallback):
    """
    Combines RewardCallback with tqdm progress bar updates.
    """
    def __init__(self, check_freq: int, pbar, n_eval_episodes: int = 5, verbose=0):
        super().__init__(check_freq, n_eval_episodes, verbose)
        self.pbar = pbar
        self.last_time_trigger = 0

    def _on_step(self) -> bool:
        # Update progress bar
        # Determine how many steps passed since last update
        # SB3 updates num_timesteps by n_envs per step call
        current_steps = self.model.num_timesteps
        delta = current_steps - self.last_time_trigger
        if delta > 0:
            self.pbar.update(delta)
            self.last_time_trigger = current_steps
            
        return super()._on_step()

