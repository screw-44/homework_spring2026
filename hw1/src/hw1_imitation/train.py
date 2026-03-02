"""Train and evaluate a Push-T imitation policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
import wandb
from torch.utils.data import DataLoader

from hw1_imitation.data import (
    Normalizer,
    PushtChunkDataset,
    download_pusht,
    load_pusht_zarr,
)
from hw1_imitation.model import build_policy, PolicyType, BasePolicy
from hw1_imitation.evaluation import Logger, evaluate_policy


LOGDIR_PREFIX = "exp"


def log_action_prediction_grid(
    model: BasePolicy,
    states_batch: torch.Tensor,
    actions_batch: torch.Tensor,
    logger: Logger,
    step: int,
    flow_num_steps: int = 10,
) -> None:
    """Plot GT vs model-predicted action curves for the first 8 samples in the batch.

    Creates a 4-row × 2-col grid of subplots (one per sample), each showing
    the GT action trajectory and the predicted one.  The figure caption
    includes the mean MSE across the 8 samples.

    Args:
        model: policy (will be temporarily switched to eval mode).
        states_batch: ``[batch, state_dim]`` on the model's device.
        actions_batch: ``[batch, chunk_size, action_dim]``.
        logger: Logger instance used to push the image to WandB.
        step: current global training step.
        flow_num_steps: Euler steps for flow-based policies.
    """
    N = min(4, states_batch.size(0))
    gt = actions_batch[:N]  # [N, chunk_size, action_dim]

    model.eval()
    with torch.no_grad():
        pred = model.sample_actions(states_batch, num_steps=flow_num_steps)  # [batch, chunk_size, action_dim], beast不能切除后再压缩
    model.train()

    pred = pred[:N]

    mse_per_sample = ((pred - gt) ** 2).mean(dim=(1, 2))  # [N]
    mean_mse = mse_per_sample.mean().item()

    gt_np   = gt.cpu().numpy()    # [N, T, D]
    pred_np = pred.cpu().numpy()  # [N, T, D]
    T = gt_np.shape[1]
    t = np.arange(T)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        if i >= N:
            ax.axis("off")
            continue
        for d in range(gt_np.shape[2]):
            ax.plot(t, gt_np[i, :, d],   linestyle="--", alpha=0.7, label=f"GT dim{d}")
            ax.plot(t, pred_np[i, :, d], linestyle="-",  alpha=0.9, label=f"Pred dim{d}")
        ax.set_title(f"sample {i}  MSE={mse_per_sample[i].item():.4f}", fontsize=8)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)

    fig.suptitle(f"step={step}  mean MSE (4 samples) = {mean_mse:.4f}", fontsize=10)
    fig.tight_layout()

    logger.log({"train/action_pred_grid": wandb.Image(fig)}, step=step)
    plt.close(fig)


@dataclass
class TrainConfig:
    # The path to download the Push-T dataset to.
    data_dir: Path = Path("data")

    # The policy type -- either MSE or flow.
    policy_type: PolicyType = "mse"
    # The number of denoising steps to use for the flow policy (has no effect for the MSE policy).
    flow_num_steps: int = 10
    # The action chunk size.
    chunk_size: int = 8

    # For exp2 and exp3: the chunk size the model actually works at.
    # The data is linearly interpolated to this size during training and back
    # to chunk_size during inference.  Ignored for other policy types.
    after_scale_chunk_size: int | None = None
    # For exp3_2_low_pass_flow: the kernel size of the low-pass Conv1d filter.
    # Ignored for other policy types.
    kernel_size: int | None = None
    # For exp3_3_A_free_knot_flow: enumeration step size for free-knot candidates.
    free_knot_sample_distance: float = 1.0
    # For exp3_3_A_free_knot_flow: whether all dims share a single knot vector.
    free_knot_joint_knot: bool = True
    # For exp3_3_C_VAE_flow: KL weight in β-VAE loss.
    vae_beta: float = 0.05
    # For exp3_3_D_VQ_VAE_flow: number of codebook entries.
    vq_codebook_size: int = 512
    # For exp3_3_D_VQ_VAE_flow: commitment loss coefficient.
    vq_commitment_cost: float = 0.25

    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.0
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    # The number of epochs to train for.
    num_epochs: int = 1000
    # How often to run evaluation, measured in training steps.
    eval_interval: int = 20_000
    num_video_episodes: int = 5
    video_size: tuple[int, int] = (256, 256)
    # How often to log training metrics, measured in training steps.
    log_interval: int = 100
    # Random seed.
    seed: int = 42
    # WandB project name.
    wandb_project: str = "hw1-imitation"
    # Experiment name suffix for logging and WandB.
    exp_name: str | None = None


def parse_train_config(
    args: list[str] | None = None,
    *,
    defaults: TrainConfig | None = None,
    description: str = "Train a Push-T MLP policy.",
) -> TrainConfig:
    defaults = defaults or TrainConfig()
    return tyro.cli(
        TrainConfig,
        args=args,
        default=defaults,
        description=description,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    zarr_path = download_pusht(config.data_dir)
    states, actions, episode_ends = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)

    dataset = PushtChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=config.chunk_size,
        normalizer=normalizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = build_policy(
        config.policy_type,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
        after_scale_chunk_size=config.after_scale_chunk_size,
        kernel_size=config.kernel_size,
        free_knot_sample_distance=config.free_knot_sample_distance,
        free_knot_joint_knot=config.free_knot_joint_knot,
        vae_beta=config.vae_beta,
        vq_codebook_size=config.vq_codebook_size,
        vq_commitment_cost=config.vq_commitment_cost,
    ).to(device)

    exp_name = f"seed_{config.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if config.exp_name is not None:
        exp_name += f"_{config.exp_name}"
    log_dir = Path(LOGDIR_PREFIX) / exp_name
    wandb.init(
        project=config.wandb_project, config=config_to_dict(config), name=exp_name
    )
    logger = Logger(log_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    model.train()
    global_step = 0
    for epoch in range(config.num_epochs):
        for batch in loader:
            states_batch, actions_batch = batch
            states_batch = states_batch.to(device)
            actions_batch = actions_batch.to(device)

            loss = model.compute_loss(states_batch, actions_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % config.log_interval == 0:
                logger.log({"train/loss": loss.item()}, step=global_step)
                print(f"Epoch {epoch}/{config.num_epochs}, Step {global_step}, Loss: {loss.item():.4f}")

            if global_step % config.eval_interval == 0:
                log_action_prediction_grid(
                    model=model,
                    states_batch=states_batch,
                    actions_batch=actions_batch,
                    logger=logger,
                    step=global_step,
                    flow_num_steps=config.flow_num_steps,
                )
                evaluate_policy(
                    model=model,
                    normalizer=normalizer,
                    device=device,
                    chunk_size=config.chunk_size,
                    video_size=config.video_size,
                    num_video_episodes=config.num_video_episodes,
                    flow_num_steps=config.flow_num_steps,
                    step=global_step,
                    logger=logger
                )
                model.train()

    # final evaluation
    evaluate_policy(
        model=model,
        normalizer=normalizer,
        device=device,
        chunk_size=config.chunk_size,
        video_size=config.video_size,
        num_video_episodes=config.num_video_episodes,
        flow_num_steps=config.flow_num_steps,
        step=global_step,
        logger=logger
    )
    logger.dump_for_grading()


def main() -> None:
    config = parse_train_config()
    run_training(config)


if __name__ == "__main__":
    main()
