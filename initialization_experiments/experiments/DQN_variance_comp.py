import sys
import os
import time
import json
import numpy as np
import torch as th
import torch.nn.functional as F
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt

try:
    import envpool
except ImportError:
    raise ImportError("pip install envpool")

from stable_baselines3 import DQN as VanillaDQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

os.environ["OMP_NUM_THREADS"] = "1"
th.backends.cuda.matmul.allow_tf32 = True 
th.backends.cudnn.benchmark = True 
device = "cuda" if th.cuda.is_available() else "cpu"

print(f"=== STRESS TEST FINAL: N=5 | Hard Updates | High LR | Variance Plot ===")

# --- WRAPPER ---
class EnvPoolSB3Wrapper(VecEnv):
    def __init__(self, venv):
        num_envs = venv.spec.config.num_envs
        super().__init__(num_envs, venv.observation_space, venv.action_space)
        self.venv = venv
        self.episode_returns = np.zeros(num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int32)
    
    def reset(self):
        obs, _ = self.venv.reset()
        self.episode_returns.fill(0)
        self.episode_lengths.fill(0)
        return obs
    def step_async(self, actions): self.actions = actions
    def step_wait(self):
        obs, rewards, terms, truncs, infos = self.venv.step(self.actions)
        dones = terms | truncs
        raw_rewards = infos['reward'] if 'reward' in infos else rewards
        self.episode_returns += raw_rewards
        self.episode_lengths += 1
        new_infos = []
        for i in range(self.num_envs):
            info = {}
            if dones[i]:
                info["episode"] = {"r": float(self.episode_returns[i]), "l": int(self.episode_lengths[i])}
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
            else:
                info["episode"] = None
            new_infos.append(info)
        return obs, rewards, dones, new_infos
    def close(self): self.venv.close()
    def get_attr(self, attr_name, indices=None): return [getattr(self.venv, attr_name)] * self.num_envs
    def set_attr(self, attr_name, value, indices=None): pass
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs): return [getattr(self.venv, method_name)(*method_args, **method_kwargs)] * self.num_envs
    def env_is_wrapped(self, wrapper_class, indices=None): return [False] * self.num_envs

# --- BUFFER ---
class SequenceBuffer(ReplayBuffer):
    def __init__(self, *args, n_step=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_step = n_step
        self.device_stored = th.device("cuda") 
    def sample_sequence(self, batch_size):
        upper_bound = self.buffer_size if self.full else self.pos
        upper_bound = upper_bound - self.n_step 
        start_inds = np.random.randint(0, upper_bound, size=batch_size)
        env_inds = np.random.randint(0, self.n_envs, size=(batch_size, 1))
        seq_inds = start_inds[:, None] + np.arange(self.n_step)[None, :]
        return {
            "obs": th.as_tensor(self.observations[seq_inds, env_inds], device=self.device_stored),
            "actions": th.as_tensor(self.actions[seq_inds, env_inds], device=self.device_stored),
            "rewards": th.as_tensor(self.rewards[seq_inds, env_inds], device=self.device_stored),
            "next_obs": th.as_tensor(self.next_observations[seq_inds, env_inds], device=self.device_stored),
            "dones": th.as_tensor(self.dones[seq_inds, env_inds], device=self.device_stored),
        }

# --- AGENT ---
class AggressiveDQN(VanillaDQN):
    def __init__(self, policy, env, n_step=5, lambda_coef=1.0, **kwargs):
        super().__init__(policy, env, **kwargs)
        self.n_step = n_step
        self.lambda_coef = lambda_coef
        
        if hasattr(th, "compile"):
            self.q_net = th.compile(self.q_net)
            self.q_net_target = th.compile(self.q_net_target)
            
        self.replay_buffer = SequenceBuffer(
            self.buffer_size, self.observation_space, self.action_space,
            device=self.device, n_envs=self.n_envs, n_step=self.n_step, handle_timeout_termination=False
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        scaler = th.cuda.amp.GradScaler()
        
        for _ in range(gradient_steps):
            batch = self.replay_buffer.sample_sequence(batch_size)
            
            with th.autocast(device_type='cuda', dtype=th.float16):
                with th.no_grad():
                    b, n = batch["next_obs"].shape[:2]
                    flat_next = batch["next_obs"].reshape(-1, *batch["next_obs"].shape[2:])
                    
                    target_online_q = self.q_net(flat_next)
                    best_acts = target_online_q.argmax(dim=1).unsqueeze(1)
                    target_target_q = self.q_net_target(flat_next).gather(1, best_acts).squeeze(1)
                    v_next_seq = target_target_q.reshape(b, n)
                    
                    running_target = v_next_seq[:, -1]
                    for k in reversed(range(self.n_step)):
                        next_val = v_next_seq[:, k] if k < self.n_step - 1 else v_next_seq[:, -1]
                        target_mix = (1 - self.lambda_coef) * next_val + self.lambda_coef * running_target
                        running_target = batch["rewards"][:, k] + (1.0 - batch["dones"][:, k]) * self.gamma * target_mix
                        
                    final_target = running_target.unsqueeze(1)

                current_q = self.q_net(batch["obs"][:, 0]).gather(1, batch["actions"][:, 0].long())
                loss = F.smooth_l1_loss(current_q, final_target)
            
            self.policy.optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            scaler.step(self.policy.optimizer)
            scaler.update()
            
            self._n_updates += 1
            if self._n_updates % self.target_update_interval == 0:
                self.q_net_target.load_state_dict(self.q_net.state_dict())

class FastHistoryCallback(BaseCallback):
    def __init__(self, total_timesteps, check_freq=5000): 
        super().__init__(0)
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq
        self.ep_rewards = deque(maxlen=100)
        self.data_log = []
        self.last_log_step = 0
        self.start_time = time.time()

    def _on_step(self) -> bool:
        for info in self.locals['infos']:
            if info and "episode" in info and info["episode"] is not None:
                r = info["episode"]["r"]
                if not np.isnan(r): self.ep_rewards.append(r)
        
        if self.num_timesteps - self.last_log_step >= self.check_freq:
            avg_rew = np.mean(self.ep_rewards) if len(self.ep_rewards) > 0 else -21.0
            self.data_log.append((self.num_timesteps, avg_rew))
            elapsed = time.time() - self.start_time
            fps = self.num_timesteps / (elapsed + 1e-6)
            print(f"\rStep: {self.num_timesteps/1e6:.2f}M | Rew: {avg_rew:6.1f} | FPS: {fps:4.0f} ", end="")
            self.last_log_step = self.num_timesteps
        return True

# --- MAIN TRAINING FUNCTION ---
def run_stress_test():
    NUM_ENVS = 64
    TOTAL_STEPS = 5_000_000 
    
    # AGGRESSIVE PARAMS
    BATCH_SIZE = 64 
    TRAIN_FREQ = 16 
    GRADIENT_STEPS = 16 
    LR = 5e-4 
    
    tasks = [
        {"id": "Pong-v5",          "gym_id": "PongNoFrameskip-v4"},
        {"id": "Breakout-v5", "gym_id": "BreakoutNoFrameskip-v4"},
        {"id": "SpaceInvaders-v5", "gym_id": "SpaceInvadersNoFrameskip-v4"},
    ]
    
    configs = [
        {"name": "1-Step DQN",      "n": 1, "lam": 0.0, "color": "blue"},
        {"name": "3-Step DQN",      "n": 3, "lam": 1.0, "color": "black"},
        {"name": "5-Step DQN",      "n": 5, "lam": 1.0, "color": "green"},
        {"name": "GAE (N=3, L=0.7)", "n": 3, "lam": 0.7, "color": "gray"},
        {"name": "GAE (N=5, L=0.8)", "n": 5, "lam": 0.8, "color": "crimson"},
    ]
    
    results = {}
    
    for task in tasks:
        env_id = task["id"]
        print(f"\n\n{'='*50}")
        print(f"TASK: {env_id} | Hard Updates + High LR")
        print(f"{'='*50}")
        
        vec_env = envpool.make_gymnasium(env_id, num_envs=NUM_ENVS, seed=42, stack_num=4, 
                                         episodic_life=True, reward_clip=True)
        vec_env = EnvPoolSB3Wrapper(vec_env)
        
        task_results = {}
        for cfg in configs:
            print(f"\n>>> Running: {cfg['name']}")
            vec_env.reset()
            th.cuda.empty_cache()
            
            model = AggressiveDQN(
                "CnnPolicy", vec_env,
                n_step=cfg["n"], 
                lambda_coef=cfg["lam"],
                learning_rate=LR, 
                buffer_size=100_000, 
                learning_starts=25_000, 
                batch_size=BATCH_SIZE,
                train_freq=TRAIN_FREQ,       
                gradient_steps=GRADIENT_STEPS,   
                target_update_interval=1000, # Hard Update
                exploration_fraction=0.1, 
                exploration_final_eps=0.01,
                gamma=0.99, verbose=0, device="cuda"
            )
            
            logger = FastHistoryCallback(total_timesteps=TOTAL_STEPS, check_freq=50000)
            model.learn(total_timesteps=TOTAL_STEPS, callback=logger, progress_bar=False)
            task_results[cfg['name']] = logger.data_log
            del model
            
        results[task["gym_id"]] = task_results
        vec_env.close()

    return results, configs

# --- PLOTTING FUNCTIONS ---
def plot_poster(results, configs):
    for env_name, res in results.items():
        plt.figure(figsize=(12, 7))
        for cfg in configs:
            name = cfg["name"]
            if name in res:
                data = np.array(res[name])
                if len(data) > 0:
                    x, y = data[:, 0], data[:, 1]
                    plt.plot(x, y, label=name, color=cfg["color"], linewidth=2, alpha=cfg["alpha"])

        plt.title(f"Stress Test (Hard Update): {env_name}", fontsize=16)
        plt.xlabel("Timesteps")
        plt.ylabel("Reward")
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{env_name}_stress.png", dpi=150)
        print(f"\nSaved {env_name}_stress.png")

def plot_with_variance(results, configs):
    WINDOW = 5
    for env_name, res in results.items():
        fig, ax = plt.subplots(figsize=(12, 8))
        for cfg in configs:
            name = cfg["name"]
            if name in res:
                raw_data = np.array(res[name])
                if len(raw_data) < WINDOW + 1: continue
                x = raw_data[:, 0]
                y = raw_data[:, 1]
                
                kernel = np.ones(WINDOW) / WINDOW
                y_mean = np.convolve(y, kernel, mode='valid')
                x_mean = x[WINDOW-1:]
                
                y_std = []
                for i in range(len(y) - WINDOW + 1):
                    y_std.append(np.std(y[i : i + WINDOW]))
                y_std = np.array(y_std)
                
                ax.plot(x_mean, y_mean, label=name, color=cfg["color"], linewidth=2.5)
                ax.fill_between(x_mean, y_mean - y_std, y_mean + y_std, color=cfg["color"], alpha=0.25)

        ax.set_title(f"Stability (Variance) Analysis: {env_name}", fontsize=16)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Reward")
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{env_name}_variance.png", dpi=150)
        print(f"Saved {env_name}_variance.png")

def save_results(results, filename="results_data.json"):
    serializable_results = {}
    for env_key, env_val in results.items():
        serializable_results[env_key] = {}
        for algo_key, algo_val in env_val.items():
            serializable_results[env_key][algo_key] = np.array(algo_val).tolist()
    with open(filename, 'w') as f:
        json.dump(serializable_results, f)
    print(f"Data saved to {filename}")

# --- EXECUTION ---
if __name__ == "__main__":
    data, confs = run_stress_test()
    save_results(data)
    plot_poster(data, confs)
    plot_with_variance(data, confs)
