import sys
import os

# Ограничиваем потоки (чтобы CPU не захлебнулся)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sys.modules["rich"] = None 

import gymnasium as gym
import torch as th
import numpy as np
from stable_baselines3 import DQN as VanillaDQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from collections import deque
import matplotlib.pyplot as plt
import time
import gc
from torch.nn import functional as F

device = "cuda" if th.cuda.is_available() else "cpu"
print(f"=== FINAL BACKUP: 4 FAST & SAFE ENVIRONMENTS ===")
print(f"Device: {device.upper()}")

# --- 1. BUFFER ---
class VectorizedSequenceBuffer(ReplayBuffer):
    def __init__(self, *args, n_step=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_step = n_step

    def sample_sequence(self, batch_size):
        upper_bound = self.buffer_size if self.full else self.pos
        start_inds = np.random.randint(0, upper_bound - self.n_step, size=batch_size)
        env_inds = np.random.randint(0, self.n_envs, size=(batch_size, 1))
        
        seq_inds = start_inds[:, None] + np.arange(self.n_step)[None, :]
        
        obs = self.observations[seq_inds, env_inds, :]      
        actions = self.actions[seq_inds, env_inds, :]       
        rewards = self.rewards[seq_inds, env_inds]          
        next_obs = self.next_observations[seq_inds, env_inds, :] 
        dones = self.dones[seq_inds, env_inds]              

        return {
            "obs": th.as_tensor(obs).to(device),
            "actions": th.as_tensor(actions).to(device),
            "rewards": th.as_tensor(rewards).to(device),
            "next_obs": th.as_tensor(next_obs).to(device),
            "dones": th.as_tensor(dones).to(device),
        }

# --- 2. AGENT ---
class GAEDQN_Universal(VanillaDQN):
    def __init__(self, policy, env, lambda_coef=0.5, n_step=3, use_target_net=True, **kwargs):
        super().__init__(policy, env, **kwargs)
        self.lambda_coef = lambda_coef
        self.n_step = n_step
        self.use_target_net = use_target_net
        
        self.replay_buffer = VectorizedSequenceBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            device=self.device,
            n_envs=self.n_envs,
            n_step=self.n_step
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        
        for _ in range(gradient_steps):
            batch = self.replay_buffer.sample_sequence(batch_size)
            
            obs_seq = batch["obs"]
            act_seq = batch["actions"].long()
            rew_seq = batch["rewards"]
            done_seq = batch["dones"]
            next_obs_seq = batch["next_obs"]

            with th.no_grad():
                # Универсальная обработка размерностей (Fixes unpack error)
                obs_shape = next_obs_seq.shape
                b, n = obs_shape[0], obs_shape[1]
                # Сплющиваем все остальные измерения
                flat_next = next_obs_seq.reshape(b * n, -1)
                
                flat_online = self.q_net(flat_next)
                flat_best_act = flat_online.argmax(dim=1).unsqueeze(1)
                
                target_net = self.q_net_target if self.use_target_net else self.q_net
                flat_target_vals = target_net(flat_next)
                flat_v_next = flat_target_vals.gather(1, flat_best_act).squeeze(1)
                v_next_seq = flat_v_next.reshape(b, n)

                running_target = v_next_seq[:, -1]
                for k in reversed(range(self.n_step)):
                    r = rew_seq[:, k]
                    d = done_seq[:, k]
                    v_next = v_next_seq[:, k]
                    
                    if k == self.n_step - 1:
                        target_mix = v_next
                    else:
                        target_mix = (1 - self.lambda_coef) * v_next + self.lambda_coef * running_target
                    
                    running_target = r + (1 - d) * self.gamma * target_mix
                
                final_target = running_target.unsqueeze(1)

            current_q = self.q_net(obs_seq[:, 0]).gather(1, act_seq[:, 0])
            loss = F.smooth_l1_loss(current_q, final_target)
            
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        if self.use_target_net and (self._n_updates % self.target_update_interval == 0):
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

# --- 3. LOGGER ---
class HistoryCallback(BaseCallback):
    def __init__(self, total_timesteps, check_freq=1000, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq
        self.ep_rewards = deque(maxlen=100)
        self.data_log = []
        self.last_log_step = 0

    def _on_step(self) -> bool:
        for info in self.locals['infos']:
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
        
        if self.num_timesteps - self.last_log_step >= self.check_freq:
            avg_rew = np.mean(self.ep_rewards) if len(self.ep_rewards) > 0 else -float('inf')
            self.data_log.append((self.num_timesteps, avg_rew))
            
            progress = (self.num_timesteps / self.total_timesteps) * 100
            r_str = f"{avg_rew:6.1f}" if avg_rew != -float('inf') else "Wait"
            print(f"\r   [{progress:3.0f}%] Steps: {self.num_timesteps} | Rew: {r_str} ", end="")
            self.last_log_step = self.num_timesteps
        return True

# --- 4. FACTORY (FIXED FOR RAM) ---
def create_safe_env(env_id, is_atari=False):
    """
    Создает среду гарантированно в режиме векторов (RAM).
    """
    if is_atari:
        try:
            import ale_py
            import shimmy
            gym.register_envs(ale_py)
            # Явно запрашиваем RAM. Это вернет вектор (128,), а не картинку
            env = gym.make(f"ALE/{env_id}-v5", obs_type="ram")
        except Exception as e:
            print(f"Warning: ALE/{env_id}-v5 failed ({e}), trying fallback...")
            try:
                # Fallback для старых версий
                env = gym.make(f"{env_id}-ram-v4")
            except:
                print(f"CRITICAL: Could not load {env_id}. Replacing with CartPole.")
                env = gym.make("CartPole-v1")
    else:
        env = gym.make(env_id)
        
    return Monitor(env)

# --- 5. RUNNER ---
def run_backup_final_safe():
    N_ENVS = 16 
    BATCH_SIZE = 256
    LR = 5e-4
    
    # Агрессивные обновления для быстрых сред
    TRAIN_FREQ = 1 
    GRADIENT_STEPS = 4 
    
    tasks_config = [
        # 1. CartPole (Гарантия)
        {"name": "Assault (RAM)", "id": "Assault", "atari": True, "steps": 300_000},
        {"name": "CartPole", "id": "CartPole-v1", "atari": False, "steps": 200_000},
        
        # 2. Acrobot (Физика)
        {"name": "Acrobot",  "id": "Acrobot-v1",  "atari": False, "steps": 200_000},
        
        # 3. Pong RAM (Вместо LunarLander/MountainCar)
        # RAM версия учится быстро и дает красивый график роста
        
        
        # 4. Breakout RAM (Классика)
        {"name": "Breakout (RAM)", "id": "Breakout", "atari": True, "steps": 200_000},
    ]
    
    configs = [
        {"name": "Standard DQN",      "lam": 1.0, "n": 1, "target": True,  "color": "black", "style": "--", "width": 1.5},
        {"name": "No Frozen Target",  "lam": 1.0, "n": 1, "target": False, "color": "gray",  "style": ":",  "width": 1.5},
        {"name": "N-Step (N=3)",      "lam": 1.0, "n": 3, "target": True,  "color": "green", "style": "-",  "width": 1.5},
        {"name": "N-Step (N=5)",      "lam": 1.0, "n": 5, "target": True,  "color": "teal",  "style": "-",  "width": 1.5},
        {"name": "GAE (N=3, lam=0.6)","lam": 0.85, "n": 3, "target": True,  "color": "#3498db", "style": "-", "width": 2.0},
        {"name": "GAE (N=5, lam=0.6)","lam": 0.85, "n": 5, "target": True,  "color": "crimson", "style": "-", "width": 2.5},
    ]
    
    results = {}
    
    for task in tasks_config:
        env_name = task["name"]
        print(f"\n\n--- Environment: {env_name} ---")
        
        # Создаем DummyVecEnv (Самый надежный)
        vec_env = DummyVecEnv([lambda: create_safe_env(task["id"], task["atari"]) for _ in range(N_ENVS)])
            
        task_results = {}
        
        for cfg in configs:
            print(f"\nTraining: {cfg['name']}")
            vec_env.reset()
            gc.collect()
            
            model = GAEDQN_Universal(
                "MlpPolicy", # Используем MLP, так как везде вектора (даже в Pong RAM)
                vec_env,
                lambda_coef=cfg["lam"],
                n_step=cfg["n"],
                use_target_net=cfg["target"],
                learning_rate=LR,
                buffer_size=100_000,
                learning_starts=2000,
                batch_size=BATCH_SIZE,
                train_freq=TRAIN_FREQ,       
                gradient_steps=GRADIENT_STEPS,   
                gamma=0.99,
                target_update_interval=500,
                verbose=0,
                device=device
            )
            
            steps = task["steps"]
            log_freq = max(500, steps // 100)
            logger = HistoryCallback(total_timesteps=steps, check_freq=log_freq)
            
            model.learn(total_timesteps=steps, callback=logger, progress_bar=False)
            task_results[cfg['name']] = logger.data_log
            del model
            
        results[env_name] = task_results
        vec_env.close()

    return results, configs

def plot_backup_results(results, configs, filename="backup_poster_final.png"):
    env_ids = list(results.keys())
    n_envs = len(env_ids)
    
    if n_envs == 0: return

    cols = min(n_envs, 2)
    rows = (n_envs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n_envs == 1: axes = [axes]
    else: axes = axes.flatten()
    
    for idx, env_id in enumerate(env_ids):
        ax = axes[idx]
        env_data = results[env_id]
        
        for cfg in configs:
            name = cfg["name"]
            color = cfg["color"]
            style = cfg.get("style", "-")
            width = cfg.get("width", 2.0)
            
            if name in env_data:
                history = np.array(env_data[name])
                if len(history) == 0: continue
                
                steps = history[:, 0]
                rewards = history[:, 1]
                
                # Сглаживание
                window = 10
                if len(rewards) >= window:
                    smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    smoothed_steps = steps[window-1:]
                else:
                    smoothed_rewards = rewards
                    smoothed_steps = steps
                
                ax.plot(smoothed_steps, smoothed_rewards, label=name, color=color, linestyle=style, linewidth=width)
        
        ax.set_title(f"{env_id}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Timesteps", fontsize=12)
        ax.set_ylabel("Reward", fontsize=12)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    print(f"\n\nSaving backup plot to {filename}...")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    res, confs = run_backup_final_safe()
    plot_backup_results(res, confs)
