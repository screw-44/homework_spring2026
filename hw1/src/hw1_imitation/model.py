"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn
from transformers import AutoProcessor


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

class Exp2SparseFlowMatchingPolicy(BasePolicy):
    """Flow matching policy that operates at a rescaled chunk resolution.

    The network works internally at ``after_scale_chunk_size`` resolution.
    - Training:  action chunk (``chunk_size``) is interpolated to
      ``after_scale_chunk_size`` before computing the flow matching loss.
    - Inference: actions are generated at ``after_scale_chunk_size`` and
      interpolated back to ``chunk_size`` before returning.

    This lets you study whether *point count* or *represented time span*
    drives policy performance.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        after_scale_chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.after_scale_chunk_size = after_scale_chunk_size

        # Network input: state + flat actions at scaled resolution + timestep
        flat_scaled_dim = after_scale_chunk_size * action_dim
        layers: list[nn.Module] = []
        in_dim = state_dim + flat_scaled_dim + 1
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, flat_scaled_dim))
        self.net = nn.Sequential(*layers)

        print(f"Exp2SparseFlowMatchingPolicy: chunk_size={chunk_size} -> after_scale={after_scale_chunk_size}, \n self.net:{self.net}")

    def _interp(self, actions: torch.Tensor, target_len: int) -> torch.Tensor:
        """Linearly interpolate ``[batch, src_len, action_dim]`` to ``target_len``."""
        if actions.shape[1] == target_len:
            return actions
        # F.interpolate expects (N, C, L)
        x = actions.permute(0, 2, 1)
        x = nn.functional.interpolate(x, size=target_len, mode="linear", align_corners=True)
        return x.permute(0, 2, 1)

    def compute_loss(
        self,
        state: torch.Tensor,        # [batch, state_dim]
        action_chunk: torch.Tensor, # [batch, chunk_size, action_dim]
    ) -> torch.Tensor:
        batch_size = state.size(0)
        # Scale ground-truth chunk to the internal resolution
        x_1 = self._interp(action_chunk, self.after_scale_chunk_size)
        x_1 = x_1.reshape(batch_size, -1)  # [batch, flat_scaled_dim]

        x_0 = torch.randn_like(x_1)
        t = torch.rand(batch_size, 1, device=state.device)
        x_t = (1.0 - t) * x_0 + t * x_1
        u_t = x_1 - x_0  # target velocity

        pred_u_t = self.net(torch.cat([state, x_t, t], dim=-1))
        return nn.functional.mse_loss(pred_u_t, u_t)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.size(0)
        flat_scaled_dim = self.after_scale_chunk_size * self.action_dim

        # Euler integration at the internal scaled resolution
        x_t = torch.randn(batch_size, flat_scaled_dim, device=state.device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((batch_size, 1), i * dt, device=state.device)
            x_t = x_t + self.net(torch.cat([state, x_t, t], dim=-1)) * dt

        scaled = x_t.reshape(batch_size, self.after_scale_chunk_size, self.action_dim)
        # Interpolate back to the real chunk_size
        return self._interp(scaled, self.chunk_size)

class Exp3BeastFlowMatchingPolicy(BasePolicy):
    """Flow matching policy that uses B-spline (BEAST) encoding as the latent space.

    The network operates in the BEAST-encoded space of size
    ``after_scale_chunk_size * action_dim`` (B-spline coefficients, normalized to [-1, 1]).
    - Training:  action chunk → B-spline encode → flow matching loss in latent space.
    - Inference: Euler integration in latent space → B-spline decode → action chunk.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        after_scale_chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.after_scale_chunk_size = after_scale_chunk_size

        #@ 网上的版本包含了normalization
        # self.beast = AutoProcessor.from_pretrained(
        #     "zhouhongyi/beast",
        #     trust_remote_code=True,
        #     num_dof=action_dim,
        #     num_basis=after_scale_chunk_size,
        #     seq_len=chunk_size,
        #     degree_p=3,
        #     device="cuda",
        # )

        #@ 这个版本不包含normalization，和直接插值更公平的对比
        from hw1_imitation.compressor.beast import BeastTokenizer
        self.beast = BeastTokenizer(
            num_dof = action_dim,
            num_basis = after_scale_chunk_size,
            seq_len = chunk_size,
            degree_p = 3,
            device = 'cuda'
        )

        # Same network structure as Exp2SparseFlowMatchingPolicy,
        # but the latent dimension is after_scale_chunk_size * action_dim (B-spline coefficients).
        flat_latent_dim = after_scale_chunk_size * action_dim
        layers: list[nn.Module] = []
        in_dim = state_dim + flat_latent_dim + 1
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, flat_latent_dim))
        self.net = nn.Sequential(*layers)

        print(f"Exp3BeastFlowMatchingPolicy: chunk_size={chunk_size}, num_basis={after_scale_chunk_size}")

    def _encode(self, action_chunk: torch.Tensor) -> torch.Tensor:
        """B-spline encode each item in the batch.

        Args:
            action_chunk: ``[batch, chunk_size, action_dim]``
        Returns:
            ``[batch, after_scale_chunk_size * action_dim]``, normalized to [-1, 1]
        """
        return self.beast.encode_continuous(action_chunk, update_bounds=True)
  

    def _decode(self, encoded_action_chunk: torch.Tensor) -> torch.Tensor:
        """B-spline decode each item in the batch.

        Args:
            encoded_action_chunk: ``[batch, after_scale_chunk_size * action_dim]``
        Returns:
            ``[batch, chunk_size, action_dim]``
        """
        return self.beast.decode_continuous(encoded_action_chunk)

    def compute_loss(
        self,
        state: torch.Tensor,        # [batch, state_dim]
        action_chunk: torch.Tensor, # [batch, chunk_size, action_dim]
    ) -> torch.Tensor:
        batch_size = state.size(0)
        x_1 = self._encode(action_chunk)  # [batch, flat_latent_dim]

        x_0 = torch.randn_like(x_1)
        t = torch.rand(batch_size, 1, device=state.device)
        x_t = (1.0 - t) * x_0 + t * x_1
        u_t = x_1 - x_0  # target velocity

        pred_u_t = self.net(torch.cat([state, x_t, t], dim=-1))
        return nn.functional.mse_loss(pred_u_t, u_t)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.size(0)
        flat_latent_dim = self.after_scale_chunk_size * self.action_dim

        # Euler integration in B-spline latent space
        x_t = torch.randn(batch_size, flat_latent_dim, device=state.device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((batch_size, 1), i * dt, device=state.device)
            x_t = x_t + self.net(torch.cat([state, x_t, t], dim=-1)) * dt

        return self._decode(x_t)  # [batch, chunk_size, action_dim]

PolicyType: TypeAlias = Literal["mse", "flow", "exp2_sparse_flow", "exp3_beast_flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
    after_scale_chunk_size: int | None = None,
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
    if policy_type == "exp2_sparse_flow":
        if after_scale_chunk_size is None:
            raise ValueError(
                "after_scale_chunk_size must be provided for exp2_sparse_flow"
            )
        return Exp2SparseFlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            after_scale_chunk_size=after_scale_chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "exp3_beast_flow":
        if after_scale_chunk_size is None:
            raise ValueError(
                "after_scale_chunk_size must be provided for exp3_beast_flow"
            )
        return Exp3BeastFlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            after_scale_chunk_size=after_scale_chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
