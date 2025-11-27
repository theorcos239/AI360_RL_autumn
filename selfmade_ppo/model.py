import torch
from torch import nn
import numpy as np

class PPOActorCritic(nn.Module):
    def __init__(self, n_actions, stack_size=3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(stack_size, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(22528, 512),
            nn.ReLU()
        )

        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        features = self.conv(x)
        return self.actor(features), self.critic(features)
