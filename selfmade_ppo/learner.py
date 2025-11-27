import time
import random
import numpy as np
import torch
from torch import nn
import torch.distributions as dist
from torch.utils.data import DataLoader

import gymnasium as gym
from tqdm import tqdm

from rollout_buffer import RolloutBuffer
from python_dataset import PythonListDataset


class PPOLearning():
    def __init__(self, actorcritic, env, device):
        self.actorcritic = actorcritic
        self.env = env
        self.device = device
        self.version = 0

    def create_rollout(self, length = 1024):
        rollout = RolloutBuffer(device=self.device)
        obs, info = self.env.reset()

        with torch.no_grad():
            for _ in range(length):
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0).permute([0, 3, 1, 2]) / 255.0

                logits, value = self.actorcritic(obs_tensor)
                value = float(value.squeeze(0).cpu().item())

                distribution = dist.Categorical(logits=logits)
                action_t = distribution.sample()
                action = int(action_t.cpu().item())


                log_prob = float(distribution.log_prob(action_t).cpu().item())

                new_obs, reward, terminated, truncated, info = self.env.step(action)
                done = int(terminated or truncated)

                rollout.add(
                    action=action,
                    log_prob=log_prob,
                    reward=reward,
                    value=value,
                    state=obs,
                    done=done
                )

                obs = new_obs
                if done:
                    obs, info = self.env.reset()

            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0).permute([0, 3, 1, 2]) / 255.0

            _, last_value = self.actorcritic(obs_tensor)
            rollout.finish(float(last_value.squeeze(0).cpu().item()))

        return rollout

    def process_gae(self, rollout, gamma = 0.99, delta = 0.95):
      
        n_rollout = len(rollout.actions)
        est1 = []
        for i in range(n_rollout):
            v_next = rollout.values[i + 1]
            v_cur = rollout.values[i]
            est1.append(rollout.rewards[i] + (1 - rollout.done[i]) * gamma * v_next - v_cur)

        gae_est = []
        for i in reversed(range(n_rollout)):
            if i == n_rollout - 1:
                gae_est.append(est1[i])
            else:
                gae_est.append(est1[i] + (1 - rollout.done[i]) * gamma * delta * gae_est[-1])
        gae = torch.tensor(gae_est[::-1], dtype=torch.float32, device=self.device)
        return gae

    def create_dataset(self, rollout, gamma = 0.99, delta = 0.95):
        
        states, actions, rewards, values_tensor, log_probs, done = rollout.to_tensors(self.device)
        values = values_tensor[:-1]

        gae = self.process_gae(rollout, gamma, delta)
        
        gae = (gae - gae.mean()) / (gae.std(unbiased=False) + 1e-8)

        target_critic = values + gae

        dataset = list(zip(
            states,            
            actions,         
            gae,          
            target_critic,
            log_probs,
            values
        ))

        return PythonListDataset(dataset)
    
    def create_dataset_from_list_of_rollouts(self, rollouts, gamma = 0.99, delta = 0.95):
        dataset_of_list = None

        for rollout in rollouts:
            if not dataset_of_list:
                dataset_of_list = self.create_dataset(rollout, gamma, delta)
            else:
                cur_dataset = self.create_dataset(rollout, gamma, delta)
                dataset_of_list.items.extend(cur_dataset.items)
        
        return dataset_of_list

    def one_epoch_of_learning(self, rollout,
                              optimizer,
                              batch_size = 256, n_epochs = 4,
                              gamma = 0.99, delta = 0.95, clip_eps = 0.2,
                              entropy_coef = 0.01, value_coef = 0.5):
    
        if isinstance(rollout, PythonListDataset):
            dataset = rollout
        else:
            dataset = self.create_dataset(rollout, gamma, delta)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(n_epochs):
            for sample in dataloader:
                states, actions, gae, y, old_log_probs, old_values = sample

               
                if isinstance(states, torch.Tensor):
                    states = states.to(self.device).float()
                else:
                    states = torch.tensor(states, dtype=torch.float32, device=self.device)
                states = torch.tensor(states, dtype=torch.float32, device=self.device).permute([0, 3, 1, 2]) / 255.0

                actions = actions.to(self.device).long().squeeze(-1) if actions.dim() > 1 else actions.to(self.device).long()
                gae = gae.to(self.device).float().squeeze(-1) if gae.dim() > 1 else gae.to(self.device).float()
                y = y.to(self.device).float().squeeze(-1) if y.dim() > 1 else y.to(self.device).float()
                old_log_probs = old_log_probs.to(self.device).float().squeeze(-1) if old_log_probs.dim() > 1 else old_log_probs.to(self.device).float()
                old_values = old_values.to(self.device).float().squeeze(-1) if old_values.dim() > 1 else old_values.to(self.device).float()


                optimizer.zero_grad()

                new_logits, new_values = self.actorcritic(states)
                new_values = new_values.squeeze(-1)

                v_pred = new_values
                v_old = old_values
                v_clipped = v_old + (v_pred - v_old).clamp(-clip_eps, clip_eps)
                loss1 = (v_pred - y).pow(2)
                loss2 = (v_clipped - y).pow(2)
                critic_loss = 0.5 * torch.max(loss1, loss2).mean()

                distribution = dist.Categorical(logits=new_logits)
                new_log_probs = distribution.log_prob(actions)

                ratio = torch.exp(new_log_probs - old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

                loss_pi = ratio * gae
                loss_pi_clip = clipped_ratio * gae
                actor_loss = -torch.min(loss_pi, loss_pi_clip).mean()

                entropy = distribution.entropy().mean()

                total_loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actorcritic.parameters(), max_norm=0.5)
                optimizer.step()

    def train_actor_critic(self, steps = 40, lr = 3e-4, n_epochs = 4,
                           n_rollout = 1024, gamma = 0.99, delta = 0.95,
                           clip_eps = 0.2, batch_size = 256):
        optimizer = torch.optim.Adam(self.actorcritic.parameters(), lr=lr)

        for i in tqdm(range(steps)):
            rollout = self.create_rollout(length=n_rollout)
            print('rollout collect')

            self.one_epoch_of_learning(rollout,
                                       optimizer,
                                       batch_size=batch_size,
                                       n_epochs=n_epochs,
                                       gamma=gamma, delta=delta,
                                       clip_eps=clip_eps)

            test_reward = 0.0
            obs, _ = self.env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    s = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0).permute([0, 3, 1, 2]) / 255.0
                    logits, _ = self.actorcritic(s)
                    a = int(torch.argmax(logits, dim=-1).cpu().item())
                obs, r, term, trunc, _ = self.env.step(a)
                test_reward += float(r)
                done = term or trunc

            print(f"Iteration {i+1}/{steps} | Test reward = {test_reward:.1f}")
    
    def testing(self):
        reward = 0
        for _ in range(5):
            test_reward = 0.0
            obs, _ = self.env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    s = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0).permute([0, 3, 1, 2]) / 255.0
                    logits, _ = self.actorcritic(s)
                    a = int(torch.argmax(logits, dim=-1).cpu().item())
                obs, r, term, trunc, _ = self.env.step(a)
                test_reward += float(r)
                done = term or trunc
            reward += test_reward
        reward = reward / 5
            
        print('test_reward:', reward)
        return reward


