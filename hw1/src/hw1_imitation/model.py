"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        output_dim = chunk_size * action_dim
        
        layers: list[nn.Module] = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor, # [batch, state_dim]
        action_chunk: torch.Tensor, # [batch, chunk_size, action_dim]
    ) -> torch.Tensor:
        pred = self.net(state)
        target = action_chunk.view(action_chunk.shape[0], -1) # flatten target
        return nn.functional.mse_loss(pred, target)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        pred = self.net(state)
        return pred.view(pred.shape[0], self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss.
    
    Conditional Flow Matching (CFM) with a linear interpolation path:
        x_t = (1-t) * x_0 + t * x_1
    where x_0 ~ N(0, I) and x_1 is the target actio chunk.
    The target vector field is u_t = x_1 - x_0.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        flat_action_dim = chunk_size * action_dim
        input_dim = state_dim + flat_action_dim + 1 # state + action + timestep scalar

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, flat_action_dim))
        self.net = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor, # [batch, state_dim]
        action_chunk: torch.Tensor, # [batch, chunk_size, action_dim]
    ) -> torch.Tensor:
        batch_size = state.size(0)
        x_1 = action_chunk.reshape(batch_size, -1) # [batch, flat_action_dim]
        # sample x_0
        x_0 = torch.randn_like(x_1)
        # sample timestep t
        t = torch.rand(batch_size, 1, device=state.device) # [batch, 1]
        # Linear interpolation: x_t = (1-t)*x_0 + t*x_1
        x_t = (1.0 - t) * x_0 + t * x_1
        # Target verlocity
        u_t = x_1 - x_0
        # Predict velocity field
        net_input = torch.cat([state, x_t, t], dim=-1)
        pred_u_t = self.net(net_input)

        return nn.functional.mse_loss(pred_u_t, u_t)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.size(0)
        flatten_action_dim = self.chunk_size * self.action_dim

        # Start from noise x_0 ~ N(0, 1)
        x_t = torch.randn(batch_size, flatten_action_dim, device=state.device)

        # Euler integration from t=0 to t=1
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((batch_size, 1), i * dt, device=state.device)
            net_input = torch.cat([state, x_t, t], dim=-1)
            pred_u_t = self.net(net_input)
            x_t = x_t + pred_u_t * dt
        
        return x_t.reshape(batch_size, self.chunk_size, self.action_dim)

PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
