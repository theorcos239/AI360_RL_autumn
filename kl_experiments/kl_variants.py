"""
Adaptive KL Penalty for PPO: A Comparative Study

This script implements Proximal Policy Optimization (PPO) with various
strategies for adapting the KL penalty coefficient (beta) to enforce
trust region constraints.

Algorithms implemented:
1. 'clip': Standard PPO Clipping (no KL penalty).
2. 'kl_default': Traditional adaptive KL (double/halve beta).
3. 'kl_instant': Updates beta based on instantaneous mean KL per epoch.
4. 'kl_smooth': Updates beta based on Exponential Moving Average (EMA) of KL.
5. 'kl_batch_adaptive': Modulates beta per mini-batch (Proportional control).
6. 'kl_smooth_adaptive': Hybrid of 'kl_smooth' (global) and 'kl_batch_adaptive' (local).
7. 'kl_barrier': Refined hybrid with a "soft barrier" floor (Recommended).

Usage:
    python kl_smooth.py
"""

# --- INSTALL DEPENDENCIES (Uncomment if needed) ---
# !pip install dm_control mujoco torch tqdm matplotlib pandas seaborn

import os
import math
import queue
import random
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from copy import deepcopy
from dm_control import suite
from tqdm import tqdm

# --- A100 SETUP ---
torch.set_default_dtype(torch.float32)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running on: {DEVICE}")

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    'domain': 'cheetah',
    'task': 'run',
    'seed': 42,
    
    # Resources
    'num_envs': 16,
    'total_steps': 1_000_000, 
    'num_seeds': 9, # Median over 9 seeds provides robust stats
    
    # PPO Hyperparameters
    'batch_size': 2048,
    'update_every': 256,
    'epochs': 10,
    'lr': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'target_kl': 0.015,
    'init_beta': 1.0,
    
    # Algorithms to run
    'algos_to_test': ['clip', 'kl_default', 'kl_instant', 'kl_smooth', 'kl_batch_adaptive', 'kl_smooth_adaptive', 'kl_barrier']
}

# ==========================================
# 2. UTILS & ENV (Threaded)
# ==========================================
class Welford:
    """Online algorithm for computing mean and variance."""
    def __init__(self):
        self.mean = 0.0
        self.var_s = 1.0
        self.count = 0.0001
        self.M2 = 0.0

    def add(self, x):
        if isinstance(x, torch.Tensor): x = x.cpu().numpy()
        if x.ndim == 2:
            batch_count = x.shape[0]
            new_count = self.count + batch_count
            delta = x - self.mean
            self.mean += np.sum(delta, axis=0) / new_count
            self.M2 += np.sum((x - self.mean) * delta, axis=0) 
            self.count = new_count
        else:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            self.M2 += (x - self.mean) * delta
        self.var_s = self.M2 / self.count if self.count > 1 else 1.0

class ThreadedWrapper:
    """
    Multithreaded wrapper for dm_control suite environments.
    Spawns 'num_envs' threads to step environments in parallel.
    """
    def __init__(self, domain, task, num_envs, seed):
        self.num_envs = num_envs
        self.queues = [queue.Queue() for _ in range(num_envs)]
        self.results = [queue.Queue() for _ in range(num_envs)]
        self.threads = []
        self.running = True

        # Dummy env to get specs
        dummy = suite.load(domain, task)
        ospec, aspec = dummy.observation_spec(), dummy.action_spec()
        self.obs_dim = np.sum([math.prod(ospec[k].shape) for k in ospec])
        self.act_dim = math.prod(aspec.shape)
        dummy.close()

        for i in range(num_envs):
            t = threading.Thread(target=self._worker, args=(i, domain, task, seed + i))
            t.daemon = True
            t.start()
            self.threads.append(t)

    def _process(self, ts):
        o = np.array([], dtype=np.float32)
        for k in ts.observation:
            v = ts.observation[k]
            o = np.concatenate((o, v.flatten() if v.shape else [v]))
        return o.astype(np.float32), ts.reward or 0.0, ts.last()

    def _worker(self, idx, domain, task, seed):
        env = suite.load(domain, task, task_kwargs={'random': seed})
        while self.running:
            try:
                cmd, data = self.queues[idx].get(timeout=0.1)
            except: continue
            if cmd == 'step':
                ts = env.step(data)
                o, r, d = self._process(ts)
                if d: ts = env.reset(); o, _, _ = self._process(ts)
                self.results[idx].put((o, r, d))
            elif cmd == 'reset':
                ts = env.reset(); o, r, d = self._process(ts)
                self.results[idx].put(o)
            elif cmd == 'close':
                env.close(); break

    def step(self, acts):
        for i, a in enumerate(acts): self.queues[i].put(('step', a))
        res = [self.results[i].get() for i in range(self.num_envs)]
        return [np.stack(x) for x in zip(*res)]

    def reset(self):
        for i in range(self.num_envs): self.queues[i].put(('reset', None))
        return np.stack([self.results[i].get() for i in range(self.num_envs)])

    def close(self):
        self.running = False
        for i in range(self.num_envs): self.queues[i].put(('close', None))
        for t in self.threads: t.join()

# ==========================================
# 3. NETWORKS
# ==========================================
def init_weights(m, gain):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None: m.bias.data.fill_(0.0)

class Actor(nn.Module):
    def __init__(self, o_dim, a_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(o_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, a_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(a_dim))
        self.net.apply(lambda m: init_weights(m, np.sqrt(2)))
        init_weights(self.net[-1], 0.01)

    def forward(self, x):
        return D.Normal(self.net(x), torch.exp(self.log_std))

class Critic(nn.Module):
    def __init__(self, o_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(o_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 1)
        )
        self.net.apply(lambda m: init_weights(m, np.sqrt(2)))
        init_weights(self.net[-1], 1.0)
    
    def forward(self, x): return self.net(x).squeeze(-1)

# ==========================================
# 4. TRAINER
# ==========================================
def train_agent(algo_name, cfg):
    """
    Main training loop for a single agent.
    
    Args:
        algo_name (str): The name of the algorithm to use (e.g. 'kl_barrier').
        cfg (dict): Configuration dictionary.
        
    Returns:
        dict: Training history (rewards, kl, beta, steps).
    """
    print(f"\n⚡ Starting training for: {algo_name.upper()}")
    
    envs = ThreadedWrapper(cfg['domain'], cfg['task'], cfg['num_envs'], cfg['seed'])
    actor = Actor(envs.obs_dim, envs.act_dim).to(DEVICE)
    critic = Critic(envs.obs_dim).to(DEVICE)
    opt_a = torch.optim.Adam(actor.parameters(), lr=cfg['lr'])
    opt_c = torch.optim.Adam(critic.parameters(), lr=cfg['lr'])
    obs_stats = Welford()
    
    beta = cfg['init_beta']
    kl_smooth = cfg['target_kl']
    
    history = {'reward': [], 'kl': [], 'beta': [], 'steps': []}
    
    obs = envs.reset()
    
    pbar = tqdm(total=cfg['total_steps'], desc=algo_name, leave=True, dynamic_ncols=True)
    global_steps = 0
    
    while global_steps < cfg['total_steps']:
        b_o, b_a, b_logp, b_r, b_v, b_d = [], [], [], [], [], []
        
        for _ in range(cfg['update_every']):
            obs_stats.add(obs)
            mean = torch.tensor(obs_stats.mean, device=DEVICE, dtype=torch.float32)
            std = torch.sqrt(torch.tensor(obs_stats.var_s, device=DEVICE, dtype=torch.float32))
            
            obs_tensor = torch.tensor(obs, device=DEVICE, dtype=torch.float32)
            obs_norm = torch.clamp((obs_tensor - mean) / (std + 1e-5), -10, 10)
            
            with torch.no_grad():
                dist = actor(obs_norm)
                val = critic(obs_norm)
                act = dist.sample()
                logp = dist.log_prob(act).sum(-1)
            
            next_obs, rew, done = envs.step(act.cpu().numpy())
            
            b_o.append(obs_norm)
            b_a.append(act)
            b_logp.append(logp)
            b_v.append(val)
            b_r.append(torch.tensor(rew, device=DEVICE, dtype=torch.float32))
            b_d.append(torch.tensor(done, device=DEVICE, dtype=torch.float32))
            
            obs = next_obs
        
        global_steps += cfg['update_every'] * cfg['num_envs']
        pbar.update(cfg['update_every'] * cfg['num_envs'])
        
        # --- GAE ---
        with torch.no_grad():
            next_mean = torch.tensor(obs_stats.mean, device=DEVICE, dtype=torch.float32)
            next_std = torch.sqrt(torch.tensor(obs_stats.var_s, device=DEVICE, dtype=torch.float32))
            obs_next_tensor = torch.tensor(obs, device=DEVICE, dtype=torch.float32)
            next_norm = torch.clamp((obs_next_tensor - next_mean)/(next_std+1e-5), -10, 10)
            next_val = critic(next_norm)
            
        b_r, b_v, b_d = torch.stack(b_r), torch.stack(b_v), torch.stack(b_d)
        adv, ret = torch.zeros_like(b_r), torch.zeros_like(b_r)
        gae = 0
        for t in reversed(range(cfg['update_every'])):
            nv = next_val if t == cfg['update_every'] - 1 else b_v[t+1]
            nnt = 1.0 - b_d[t]
            delta = b_r[t] + cfg['gamma'] * nv * nnt - b_v[t]
            gae = delta + cfg['gamma'] * cfg['gae_lambda'] * nnt * gae
            adv[t], ret[t] = gae, gae + b_v[t]
            
        f_o = torch.stack(b_o).view(-1, envs.obs_dim)
        f_a = torch.stack(b_a).view(-1, envs.act_dim)
        f_logp = torch.stack(b_logp).view(-1)
        f_adv, f_ret = adv.view(-1), ret.view(-1)
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
        
        approx_kl_epoch = 0
        n_batches = 0
        
        for _ in range(cfg['epochs']):
            idxs = torch.randperm(f_o.size(0))
            for i in range(0, f_o.size(0), cfg['batch_size']):
                idx = idxs[i:i+cfg['batch_size']]
                dist = actor(f_o[idx])
                new_logp = dist.log_prob(f_a[idx]).sum(-1)
                ratio = torch.exp(new_logp - f_logp[idx])
                
                with torch.no_grad():
                    log_r = new_logp - f_logp[idx]
                    kl = ((torch.exp(log_r) - 1) - log_r).mean()
                    approx_kl_epoch += kl.item()
                    n_batches += 1
                
                if algo_name == 'clip':
                    surr1 = ratio * f_adv[idx]
                    surr2 = torch.clamp(ratio, 0.8, 1.2) * f_adv[idx]
                    pi_loss = -torch.min(surr1, surr2).mean()
                else:
                    eff_beta = beta
                    kl_ratio = 1.0
                    
                    if algo_name in ['kl_batch_adaptive', 'kl_all_adaptive', 'kl_smooth_adaptive', 'kl_barrier']:
                        # Calculate KL ratio for proportional control
                        kl_ratio = kl.item() / cfg['target_kl']
                        
                        if algo_name == 'kl_barrier':
                            # Barrier: Lower bound 0.5 * beta, Upper bound defined by KL violation
                            factor = max(0.5, kl_ratio)
                            eff_beta = min(beta * factor, 10000.0) 
                        else:
                            # Standard proportional control
                            eff_beta = beta * kl_ratio

                    kl_div = f_logp[idx] - new_logp
                    pi_loss = -(ratio * f_adv[idx] - eff_beta * kl_div).mean()
                
                v_loss = 0.5 * (critic(f_o[idx]) - f_ret[idx]).pow(2).mean()
                (pi_loss + v_loss).backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                opt_a.step(); opt_c.step()
                opt_a.zero_grad(); opt_c.zero_grad()
                
        mean_kl = approx_kl_epoch / max(n_batches, 1)
        target = cfg['target_kl']
        
        # --- Beta Update Logic (Global) ---
        if algo_name == 'kl_default' or algo_name == 'kl_batch_adaptive':
            # Bang-bang control
            if mean_kl > target * 1.5: beta *= 2
            elif mean_kl < target / 1.5: beta /= 2
            
        elif algo_name == 'kl_instant':
            # Instantaneous proportional
            if mean_kl > 1e-6: beta *= np.clip(mean_kl / target, 0.5, 2.0)
            
        elif algo_name == 'kl_smooth' or algo_name == 'kl_smooth_adaptive' or algo_name == 'kl_barrier':
            # Integral control (Smoothed)
            kl_smooth = 0.5 * mean_kl + 0.5 * kl_smooth
            if kl_smooth > 1e-6: beta *= np.clip(kl_smooth / target, 0.5, 2.0)
            
        elif algo_name == 'kl_all_adaptive':
             if mean_kl > 1e-6: beta *= (mean_kl / target)
            
        beta = np.clip(beta, 1e-4, 10000.0)
        
        history['reward'].append(b_r.sum(0).mean().item())
        history['kl'].append(mean_kl)
        history['beta'].append(beta)
        history['steps'].append(global_steps)
        
    envs.close()
    pbar.close()
    return history

def smooth_curve(y, box_pts=10):
    if len(y) == 0: return np.array([])
    return pd.Series(y).rolling(window=box_pts, min_periods=1, center=True).mean().values

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    LOG_FILE = 'training_logs.csv'
    logs_df = pd.DataFrame()

    # Load existing logs
    if os.path.exists(LOG_FILE):
        try:
            logs_df = pd.read_csv(LOG_FILE)
            print(f"Loaded existing logs with {len(logs_df)} rows.")
        except Exception as e:
            print(f"Error loading logs: {e}")

    # Set which algos to run. Empty list = Plotting only (if data exists)
    # algos_to_run = CONFIG['algos_to_test']  # Uncomment to run all
    algos_to_run = [] # Currently set to run nothing, just plot if main is called

    for algo in algos_to_run:
        if not logs_df.empty and 'algorithm' in logs_df.columns:
            logs_df = logs_df[logs_df['algorithm'] != algo]

        print(f"\n⚡ === Training {algo} ({CONFIG['num_seeds']} seeds) ===")
        
        for seed_idx in range(CONFIG['num_seeds']):
            current_seed = CONFIG['seed'] + seed_idx
            run_config = CONFIG.copy()
            run_config['seed'] = current_seed
            
            print(f"  > Seed {seed_idx+1}/{CONFIG['num_seeds']} (val={current_seed})")
            torch.cuda.empty_cache()
            
            stats = train_agent(algo, run_config)
            
            df = pd.DataFrame(stats)
            df['algorithm'] = algo
            df['seed'] = current_seed
            logs_df = pd.concat([logs_df, df], ignore_index=True)
            
            logs_df.to_csv(LOG_FILE, index=False)

    logs_df.to_csv(LOG_FILE, index=False)
    print(f"Logs saved to {LOG_FILE}")
    
    # Plotting code inside main (basic preview)
    # For publication quality plots, use plot_lines.py or plot_best.py
    if not logs_df.empty:
        print("Generating preview plot...")
        sns.set_theme(style="darkgrid", context="talk")
        plt.figure(figsize=(24, 12.5))

        colors = {
            'clip': 'gray', 'kl_default': 'orange', 'kl_instant': 'blue', 
            'kl_smooth': 'green', 'kl_batch_adaptive': 'red', 
            'kl_all_adaptive': 'purple', 'kl_smooth_adaptive': 'magenta', 
            'kl_barrier': 'brown'
        }

        plot_df = logs_df.copy()
        plot_df['reward_smooth'] = plot_df.groupby(['algorithm', 'seed'])['reward'].transform(lambda x: smooth_curve(x.values))

        sns.lineplot(
            data=plot_df, x='steps', y='reward_smooth', hue='algorithm', 
            palette=colors, estimator='median', errorbar=('pi', 50), linewidth=3
        )
        plt.title("Training Reward (Median ± IQR)", fontsize=24)
        plt.savefig('results_preview.png', dpi=100)
        print("Preview saved to results_preview.png")

if __name__ == "__main__":
    main()
