import torch
import numpy as np

class RolloutBuffer:
    def __init__(self, device='cpu'):
        self.device = device
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.states = []
        self.done = []

    def add(self, action, log_prob, reward, value, state, done):
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.states.append(state)
        self.done.append(int(done))

    def finish(self, last_value):
        self.values.append(float(last_value))

    def __len__(self):
        return len(self.actions)

    def to_tensors(self, device):
        states = torch.tensor(np.array(self.states), dtype=torch.float32).to(device)
        actions = torch.tensor(self.actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(device)
        values = torch.tensor(self.values, dtype=torch.float32).to(device)
        log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(device)
        done = torch.tensor(self.done, dtype=torch.float32).to(device)
        return states, actions, rewards, values, log_probs, done
