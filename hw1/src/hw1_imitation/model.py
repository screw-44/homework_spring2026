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

class Exp3_2LowPassFlowMatchingPolicy(BasePolicy):
    """Flow matching policy that operates at a rescaled chunk resolution + 低阶滤波.

    The network works internally at ``after_scale_chunk_size`` resolution.
    - Training:  action chunk (``chunk_size``) is interpolated to
      ``after_scale_chunk_size`` before computing the flow matching loss.
    - Inference: actions are generated at ``after_scale_chunk_size`` and
      interpolated back to ``chunk_size`` before returning.

    低阶滤波采用的是conv1d来进行实现，传入kernel size的参数。

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
        kernel_size: int = 5,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.after_scale_chunk_size = after_scale_chunk_size
        self.kernel_size = kernel_size # 需要测试为奇数

        # Low-pass filter: depthwise Conv1d (per-channel), same-length output.
        # Initialized as a uniform averaging kernel.
        self.low_pass = nn.Conv1d(
            in_channels=action_dim,
            out_channels=action_dim,
            kernel_size=kernel_size,
            groups=action_dim,
            padding="same",
            bias=False,
        )
        nn.init.constant_(self.low_pass.weight, 1.0 / kernel_size) # 初始化成均值滤波器
        for p in self.low_pass.parameters(): # 进行冻结
            p.requires_grad_(False)

        # Network input: state + flat actions at scaled resolution + timestep
        flat_scaled_dim = after_scale_chunk_size * action_dim
        layers: list[nn.Module] = []
        in_dim = state_dim + flat_scaled_dim + 1
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, flat_scaled_dim))
        self.net = nn.Sequential(*layers)

        print(
            f"Exp3_2LowPassFlowMatchingPolicy: chunk_size={chunk_size} -> "
            f"after_scale={after_scale_chunk_size}, kernel_size={kernel_size}\n"
            f"self.net:{self.net}"
        )

    def _interp(self, actions: torch.Tensor, target_len: int) -> torch.Tensor:
        """Linearly interpolate ``[batch, src_len, action_dim]`` to ``target_len``."""
        if actions.shape[1] == target_len:
            return actions
        # F.interpolate expects (N, C, L)
        x = actions.permute(0, 2, 1)
        x = nn.functional.interpolate(x, size=target_len, mode="linear", align_corners=True)
        return x.permute(0, 2, 1)

    def _filter(self, actions: torch.Tensor) -> torch.Tensor:
        """Low-pass filter then downsample to ``after_scale_chunk_size``.

        Steps:
        1. Smooth with a depthwise Conv1d (same padding → output length == input length).
        2. Linearly interpolate to ``after_scale_chunk_size``.

        Args:
            actions: ``[batch, chunk_size, action_dim]``
        Returns:
            ``[batch, after_scale_chunk_size, action_dim]``
        """
        # Conv1d expects (N, C, L)
        x = actions.permute(0, 2, 1)   # [batch, action_dim, chunk_size]
        x = self.low_pass(x)           # [batch, action_dim, chunk_size]  (same padding)
        x = x.permute(0, 2, 1)         # [batch, chunk_size, action_dim]
        return self._interp(x, self.after_scale_chunk_size)

    def compute_loss(
        self,
        state: torch.Tensor,        # [batch, state_dim]
        action_chunk: torch.Tensor, # [batch, chunk_size, action_dim]
    ) -> torch.Tensor:
        batch_size = state.size(0)
        # Low-pass filter then scale ground-truth chunk to the internal resolution
        x_1 = self._filter(action_chunk)
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

# Note: 构造的encode和decode的误差太大了，不进行模型训练测试了。
class Exp3_3RandomBasisFlowMatchingPolicy(BasePolicy):
    """Flow matching policy whose latent space is a random orthogonal projection.

    A fixed random orthogonal basis ``B`` of shape ``[chunk_size, after_scale_chunk_size]``
    (orthonormal columns, B^T B = I) is constructed once at init and registered
    as a non-trainable buffer.

    - Encode: z = x @ B          [batch, chunk_size, D] → [batch, after_scale, D]
    - Decode: x̂ = z @ B^T        [batch, after_scale, D] → [batch, chunk_size, D]

    Network structure is identical to Exp2SparseFlowMatchingPolicy.
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

        # Random orthogonal basis: QR decomposition of a random matrix.
        # B: [chunk_size, after_scale_chunk_size],  B^T B = I
        rand = torch.randn(chunk_size, after_scale_chunk_size)
        B, _ = torch.linalg.qr(rand)          # B: [chunk_size, after_scale_chunk_size]
        self.register_buffer("B", B)           # fixed, not a parameter

        flat_latent_dim = after_scale_chunk_size * action_dim
        layers: list[nn.Module] = []
        in_dim = state_dim + flat_latent_dim + 1
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, flat_latent_dim))
        self.net = nn.Sequential(*layers)

        print(f"Exp3_3RandomBasisFlowMatchingPolicy: chunk_size={chunk_size} -> after_scale={after_scale_chunk_size}")

    def _encode(self, actions: torch.Tensor) -> torch.Tensor:
        """Project [batch, chunk_size, action_dim] → [batch, after_scale, action_dim]."""
        return torch.einsum("btd,tk->bkd", actions, self.B)

    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Unproject [batch, after_scale, action_dim] → [batch, chunk_size, action_dim]."""
        return torch.einsum("bkd,tk->btd", latent, self.B)

    def compute_loss(
        self,
        state: torch.Tensor,        # [batch, state_dim]
        action_chunk: torch.Tensor, # [batch, chunk_size, action_dim]
    ) -> torch.Tensor:
        batch_size = state.size(0)
        x_1 = self._encode(action_chunk).reshape(batch_size, -1)  # [batch, flat_latent_dim]

        x_0 = torch.randn_like(x_1)
        t = torch.rand(batch_size, 1, device=state.device)
        x_t = (1.0 - t) * x_0 + t * x_1
        u_t = x_1 - x_0

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

        x_t = torch.randn(batch_size, flat_latent_dim, device=state.device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((batch_size, 1), i * dt, device=state.device)
            x_t = x_t + self.net(torch.cat([state, x_t, t], dim=-1)) * dt

        latent = x_t.reshape(batch_size, self.after_scale_chunk_size, self.action_dim)
        return self._decode(latent)  # [batch, chunk_size, action_dim]

class Exp4AdaptiveSplineFlowMatchingPolicy(BasePolicy):
    """ Flow matching policy that adaptively chooses B-spline control points based on the input state."""
    # 实现的难度很大，训练时间很久。先不测试
    pass

class Exp5LearnEDFlowMatchingPolicy(BasePolicy):
    """Flow matching policy with a learnable encoder/decoder compressing action space.

    - Encoder: flat action [chunk*D] → latent [after_scale*D]
    - Decoder: latent [after_scale*D] → flat action [chunk*D]
    - Flow net: operates entirely in the **latent** space.

    Training loss = flow matching loss (in latent) + reconstruction loss (decoder∘encoder ≈ id).
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
        flat_action_dim = chunk_size * action_dim
        latent_dim = after_scale_chunk_size * action_dim

        # Encoder / decoder with one hidden layer for nonlinearity
        enc_hidden = hidden_dims[0]
        self.encoder = nn.Sequential(
            nn.Linear(flat_action_dim, enc_hidden),
            nn.ReLU(),
            nn.Linear(enc_hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, enc_hidden),
            nn.ReLU(),
            nn.Linear(enc_hidden, flat_action_dim),
        )

        # Flow net: state + latent + timestep → latent velocity
        layers: list[nn.Module] = []
        in_dim = state_dim + latent_dim + 1
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)

        self.current_step = 0

        print(
            f"Exp5LearnEDFlowMatchingPolicy: chunk_size={chunk_size} -> latent={after_scale_chunk_size}"
        )

    def compute_loss(
        self,
        state: torch.Tensor,        # [batch, state_dim]
        action_chunk: torch.Tensor, # [batch, chunk_size, action_dim]
    ) -> torch.Tensor:
        batch_size = state.size(0)
        x_1_flat = action_chunk.reshape(batch_size, -1)  # [batch, flat_action_dim]

        # Encode to latent space
        z_1 = self.encoder(x_1_flat)  # [batch, latent_dim]

        # Flow matching in latent space
        z_0 = torch.randn_like(z_1)
        t = torch.rand(batch_size, 1, device=state.device)
        z_t = (1.0 - t) * z_0 + t * z_1
        u_t = z_1 - z_0  # target velocity
        pred_u_t = self.net(torch.cat([state, z_t, t], dim=-1))
        flow_loss = nn.functional.mse_loss(pred_u_t, u_t)

        # Reconstruction loss: ensure decoder can invert encoder
        recon_loss = nn.functional.mse_loss(self.decoder(z_1), x_1_flat)
        
        self.current_step += 1
        if self.current_step > 10000: # 2w是验证
            self.encoder.requires_grad_(False) # 这里不再更新encoder/decoder了
            self.decoder.requires_grad_(False)
            return flow_loss # + recon_loss
        else:
            return recon_loss  # 前期先专注于训练encoder-decoder，过渡一段时间后再加上flow loss
    
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.size(0)
        latent_dim = self.after_scale_chunk_size * self.action_dim

        # Euler integration in latent space
        z_t = torch.randn(batch_size, latent_dim, device=state.device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((batch_size, 1), i * dt, device=state.device)
            z_t = z_t + self.net(torch.cat([state, z_t, t], dim=-1)) * dt

        # Decode latent back to action space
        x_flat = self.decoder(z_t)  # [batch, chunk_size * action_dim]
        return x_flat.reshape(batch_size, self.chunk_size, self.action_dim)


PolicyType: TypeAlias = Literal[
    "mse", "flow", "exp2_sparse_flow", "exp3_beast_flow", "exp3_2_low_pass_flow", "exp3_3_random_basis_flow", "exp5_learned_ed_flow"
]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
    after_scale_chunk_size: int | None = None,
    kernel_size: int | None = None,
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
    if policy_type == "exp3_2_low_pass_flow":
        if after_scale_chunk_size is None or kernel_size is None:
            raise ValueError(
                "after_scale_chunk_size and kernel_size must be provided for exp3_2_low_pass_flow"
            )
        return Exp3_2LowPassFlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            after_scale_chunk_size=after_scale_chunk_size,
            hidden_dims=hidden_dims,
            kernel_size=kernel_size,
        )
    if policy_type == "exp3_3_random_basis_flow":
        if after_scale_chunk_size is None:
            raise ValueError(
                "after_scale_chunk_size must be provided for exp3_3_random_basis_flow"
            )
        return Exp3_3RandomBasisFlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            after_scale_chunk_size=after_scale_chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "exp5_learned_ed_flow":
        if after_scale_chunk_size is None:
            raise ValueError(
                "after_scale_chunk_size must be provided for exp5_learned_ed_flow"
            )
        return Exp5LearnEDFlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            after_scale_chunk_size=after_scale_chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
