"""Tests for Exp3_3RandomBasisFlowMatchingPolicy.

Focus: reconstruction error of the random-orthogonal-basis encode→decode round-trip
across a diverse set of curve shapes.

Run from hw1/:
    uv run python test/test_exp3_3.py
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hw1_imitation.model import Exp3_3RandomBasisFlowMatchingPolicy

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DEVICE     = torch.device("cpu")
STATE_DIM  = 5
ACTION_DIM = 2
CHUNK_SIZE = 16


# ─────────────────────────────────────────────
# Build diverse curves  [num_curves, CHUNK_SIZE, ACTION_DIM]
# ─────────────────────────────────────────────

def make_curves(chunk_size: int = CHUNK_SIZE) -> tuple[torch.Tensor, list[str]]:
    """Return (curves, names).  Each curve: [chunk_size, ACTION_DIM]."""
    t = torch.linspace(0, 1, chunk_size)           # [0, 1]
    tp = torch.linspace(0, 2 * np.pi, chunk_size)  # [0, 2π]

    curves, names = [], []

    def add(x: torch.Tensor, name: str):
        # x: [chunk_size] or [chunk_size, 2]
        if x.dim() == 1:
            x = torch.stack([x, x], dim=-1)
        curves.append(x)
        names.append(name)

    # 1. constant zero
    add(torch.zeros(chunk_size), "constant 0")
    # 2. constant 0.5
    add(torch.full((chunk_size,), 0.5), "constant 0.5")
    # 3. linear ramp [0→1]
    add(t, "linear ramp")
    # 4. linear ramp with two different dims
    curves.append(torch.stack([t, 1 - t], dim=-1))
    names.append("linear ramp / descent")
    # 5. x^2
    add(t ** 2, "x²")
    # 6. x^3
    add(t ** 3, "x³")
    # 7. sqrt(x)
    add(t.sqrt(), "√x")
    # 8. sin
    add(torch.sin(tp), "sin")
    # 9. cos
    add(torch.cos(tp), "cos")
    # 10. sin + cos on two dims
    curves.append(torch.stack([torch.sin(tp), torch.cos(tp)], dim=-1))
    names.append("sin / cos")
    # 11. high-freq sin (2 periods)
    add(torch.sin(2 * tp), "sin 2x")
    # 12. step function
    step = (t > 0.5).float()
    add(step, "step")
    # 13. triangle wave
    tri = 2 * torch.abs(t - 0.5)
    add(tri, "triangle")
    # 14. random smooth (fixed seed)
    torch.manual_seed(0)
    noise = torch.randn(chunk_size)
    # smooth with a simple box filter
    kernel = torch.ones(1, 1, 5) / 5
    smooth = torch.nn.functional.conv1d(
        noise.view(1, 1, -1), kernel, padding=2
    ).squeeze()
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-8)
    add(smooth, "random smooth")

    return torch.stack(curves, dim=0), names   # [N, chunk_size, ACTION_DIM]


# ─────────────────────────────────────────────
# TEST 1: encode→decode reconstruction
# ─────────────────────────────────────────────

def test_reconstruction(after_scale: int):
    print(f"\n{'='*60}")
    print(f"TEST: encode→decode reconstruction  (after_scale={after_scale})")
    print(f"{'='*60}")

    model = Exp3_3RandomBasisFlowMatchingPolicy(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        chunk_size=CHUNK_SIZE,
        after_scale_chunk_size=after_scale,
        hidden_dims=(64, 64),
    ).to(DEVICE)

    curves, names = make_curves(CHUNK_SIZE)    # [N, T, D]
    N = len(curves)

    with torch.no_grad():
        latent = model._encode(curves)         # [N, after_scale, D]
        recon  = model._decode(latent)         # [N, T, D]

    # Per-curve MSE
    mse_per = ((recon - curves) ** 2).mean(dim=(1, 2))   # [N]
    mae_per = (recon - curves).abs().mean(dim=(1, 2))    # [N]

    print(f"\n{'Curve':<25}  {'MSE':>10}  {'MAE':>10}")
    print("-" * 48)
    for i, name in enumerate(names):
        print(f"  {name:<23}  {mse_per[i].item():>10.6f}  {mae_per[i].item():>10.6f}")
    print("-" * 48)
    print(f"  {'MEAN':<23}  {mse_per.mean().item():>10.6f}  {mae_per.mean().item():>10.6f}")
    print(f"  {'MAX' :<23}  {mse_per.max().item():>10.6f}  {mae_per.max().item():>10.6f}")

    # ── Visualize ──────────────────────────────────────────
    ncols = 4
    nrows = (N + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8))
    t = np.linspace(0, 1, CHUNK_SIZE)

    for i, ax in enumerate(axes.flat):
        if i >= N:
            ax.axis("off")
            continue
        orig_np  = curves[i].numpy()   # [T, D]
        recon_np = recon[i].numpy()
        for d in range(ACTION_DIM):
            ax.plot(t, orig_np[:, d],  linestyle="--", alpha=0.7, label=f"orig d{d}")
            ax.plot(t, recon_np[:, d], linestyle="-",  alpha=0.9, label=f"recon d{d}")
        ax.set_title(
            f"{names[i]}\nMSE={mse_per[i].item():.5f}",
            fontsize=7,
        )
        ax.legend(fontsize=5)
        ax.tick_params(labelsize=6)

    fig.suptitle(
        f"Random-Basis encode→decode  |  chunk={CHUNK_SIZE} → latent={after_scale}\n"
        f"mean MSE={mse_per.mean().item():.5f}  max MSE={mse_per.max().item():.5f}",
        fontsize=9,
    )
    fig.tight_layout()
    out_path = os.path.join(
        os.path.dirname(__file__),
        f"random_basis_recon_scale{after_scale}.png",
    )
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nVisualization saved → {out_path}")

    return mse_per.mean().item()


# ─────────────────────────────────────────────
# TEST 2: sweep compression ratios
# ─────────────────────────────────────────────

def test_compression_sweep():
    print(f"\n{'='*60}")
    print(f"TEST: MSE vs compression ratio sweep")
    print(f"{'='*60}")

    scales = [1, 2, 4, 6, 8, 10, 12, 14, 16]
    curves, names = make_curves(CHUNK_SIZE)

    mean_mses = []
    for scale in scales:
        if scale > CHUNK_SIZE:
            continue
        model = Exp3_3RandomBasisFlowMatchingPolicy(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            chunk_size=CHUNK_SIZE,
            after_scale_chunk_size=scale,
        ).to(DEVICE)
        with torch.no_grad():
            recon = model._decode(model._encode(curves))
        mse = ((recon - curves) ** 2).mean().item()
        mean_mses.append(mse)
        print(f"  after_scale={scale:2d}  mean MSE={mse:.6f}")

    # Plot sweep curve
    valid_scales = [s for s in scales if s <= CHUNK_SIZE]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(valid_scales, mean_mses, marker="o")
    ax.set_xlabel("after_scale_chunk_size (latent dim)")
    ax.set_ylabel("mean reconstruction MSE")
    ax.set_title(f"Reconstruction error vs compression ratio  (chunk_size={CHUNK_SIZE})")
    ax.axvline(CHUNK_SIZE, color="gray", linestyle=":", label="no compression (=chunk_size)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_path = os.path.join(os.path.dirname(__file__), "random_basis_sweep.png")
    fig.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nSweep plot saved → {out_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Detailed reconstruction for three representative compression factors
    for scale in [4, 8, 12]:
        test_reconstruction(after_scale=scale)

    test_compression_sweep()

    print(f"\n{'='*60}")
    print("All tests done.")
    print(f"{'='*60}")
