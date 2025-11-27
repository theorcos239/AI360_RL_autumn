import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

class XavierPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ortho_init=False)
        self.apply(self.init_weights_xavier)

    @staticmethod
    def init_weights_xavier(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

class UniformPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ortho_init=False)
        self.apply(self.init_weights_uniform)

    @staticmethod
    def init_weights_uniform(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Random uniform initialization [-0.1, 0.1] (generic, safe for linear)
            # For CNNs usually [-0.05, 0.05] is safer, but keeping consistent for general use
            nn.init.uniform_(module.weight, -0.1, 0.1)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

class NormalPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ortho_init=False)
        self.apply(self.init_weights_normal)

    @staticmethod
    def init_weights_normal(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Random normal initialization (mean=0, std=0.1)
            nn.init.normal_(module.weight, mean=0.0, std=0.1)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

class BinaryPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ortho_init=False)
        self.apply(self.init_weights_binary)

    @staticmethod
    def init_weights_binary(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            with torch.no_grad():
                module.weight.data = torch.randint(0, 2, module.weight.size()).float()
            if module.bias is not None:
                module.bias.data.fill_(0.0)

# Specialized policies for CNNs (smaller variance)
class UniformCnnPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ortho_init=False)
        self.apply(self.init_weights_uniform)

    @staticmethod
    def init_weights_uniform(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.uniform_(module.weight, -0.05, 0.05)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

class NormalCnnPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ortho_init=False)
        self.apply(self.init_weights_normal)

    @staticmethod
    def init_weights_normal(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.05)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

