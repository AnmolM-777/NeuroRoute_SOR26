import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class DQN(nn.Module):
    """
    Deep Q-Network Model Architecture.
    Maps input state vectors (telemetry observation) to Q-values for each discrete routing action.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super(DQN, self).__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]

        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        self.feature_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, action_dim)
        self.network = nn.Sequential(self.feature_layers, self.output_layer)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

    def freeze_feature_layers(self):
        """Freezes feature extraction layers for transfer learning / fine-tuning."""
        for param in self.feature_layers.parameters():
            param.requires_grad = False

    def unfreeze_all_layers(self):
        """Unfreezes all layers."""
        for param in self.parameters():
            param.requires_grad = True


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network Architecture (2024 Production Standard).
    Splits neural network into two separate streams:
      1. Value Stream: V(s) estimating the scalar value of state s.
      2. Advantage Stream: A(s, a) estimating relative advantage of each action a.
    Combined as: Q(s, a) = V(s) + [A(s, a) - mean(A(s, a))].
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None, stream_hidden_dim: Optional[int] = None):
        super(DuelingDQN, self).__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]

        # Shared feature extraction representation
        feature_layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            feature_layers.append(nn.Linear(in_dim, h_dim))
            feature_layers.append(nn.ReLU())
            in_dim = h_dim

        self.feature_layer = nn.Sequential(*feature_layers)
        
        # Dedicated Value stream
        v_dim = stream_hidden_dim if stream_hidden_dim is not None else max(32, in_dim // 2)
        self.value_stream = nn.Sequential(
            nn.Linear(in_dim, v_dim),
            nn.ReLU(),
            nn.Linear(v_dim, 1)
        )
        
        # Dedicated Advantage stream
        a_dim = stream_hidden_dim if stream_hidden_dim is not None else max(32, in_dim // 2)
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_dim, a_dim),
            nn.ReLU(),
            nn.Linear(a_dim, action_dim)
        )

        # Backward compatibility aliases
        self.value_head = self.value_stream
        self.advantage_head = self.advantage_stream

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(state)
        values = self.value_stream(features) # Shape: (batch, 1)
        advantages = self.advantage_stream(features) # Shape: (batch, action_dim)
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        return values + (advantages - advantages.mean(dim=-1, keepdim=True))

    def freeze_feature_layers(self):
        """Freezes shared feature extraction layers for transfer learning / fine-tuning."""
        for param in self.feature_layer.parameters():
            param.requires_grad = False

    def unfreeze_all_layers(self):
        """Unfreezes all layers."""
        for param in self.parameters():
            param.requires_grad = True


class StudentDQN(nn.Module):
    """
    Lightweight Student Network for Policy Distillation & Edge Switch Deployment.
    Compresses parameter footprint by 5-10x for ultra-low-latency real-time inference.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super(StudentDQN, self).__init__()
        if hidden_dims is None:
            hidden_dims = [32, 32] # 5-10x smaller than baseline

        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

