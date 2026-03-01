"""Tests for Exp3_2LowPassFlowMatchingPolicy.

Tests:
  1. _filter() correctness: check output is not zero, smoothing works as expected.
  2. Learning test: check the model can learn and sample_actions output aligns
     with ground truth (before filtering).

Run from hw1/:
    uv run python test/test_exp3_2.py
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from hw1_imitation.model import Exp3_2LowPassFlowMatchingPolicy

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DEVICE     = torch.device("cpu")
STATE_DIM  = 5
ACTION_DIM = 2
CHUNK_SIZE = 8
AFTER_SCALE= 4
KERNEL     = 3
BATCH      = 16

def make_model(kernel_size=KERNEL):
    return Exp3_2LowPassFlowMatchingPolicy(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        chunk_size=CHUNK_SIZE,
        after_scale_chunk_size=AFTER_SCALE,
        hidden_dims=(64, 64),
        kernel_size=kernel_size,
    ).to(DEVICE)


# ═══════════════════════════════════════════
# 1. FILTER CORRECTNESS
# ═══════════════════════════════════════════

def test_filter_correctness():
    print("\n" + "="*60)
    print("TEST 1: _filter() correctness")
    print("="*60)

    model = make_model(kernel_size=KERNEL)

    # ── 1a. Pure sine wave (should be almost unchanged after filtering) ──
    t_vals = torch.linspace(0, 2 * np.pi, CHUNK_SIZE)
    # shape [BATCH, CHUNK_SIZE, ACTION_DIM]
    sine_clean = torch.stack([
        torch.sin(t_vals),
        torch.cos(t_vals),
    ], dim=-1).unsqueeze(0).expand(BATCH, -1, -1)

    filtered_clean = model._filter(sine_clean)
    print(f"\n[Clean sine]  input shape : {sine_clean.shape}")
    print(f"              output shape: {filtered_clean.shape}  (expected [{BATCH},{AFTER_SCALE},{ACTION_DIM}])")
    print(f"  input  mean={sine_clean.mean():.4f}  std={sine_clean.std():.4f}  "
          f"min={sine_clean.min():.4f}  max={sine_clean.max():.4f}")
    print(f"  output mean={filtered_clean.mean():.4f}  std={filtered_clean.std():.4f}  "
          f"min={filtered_clean.min():.4f}  max={filtered_clean.max():.4f}")
    assert not torch.isnan(filtered_clean).any(), "NaN in filtered output!"
    assert filtered_clean.abs().max() > 0.01, "Output is basically ZERO — filter bug!"

    # ── 1b. Sine + high-freq noise (output should be smoother) ──
    noise = torch.randn_like(sine_clean) * 0.5
    noisy = sine_clean + noise
    filtered_noisy = model._filter(noisy)

    # Smoothness proxy: mean absolute difference between consecutive steps
    def roughness(x):
        diffs = (x[:, 1:, :] - x[:, :-1, :]).abs()
        return diffs.mean().item()

    r_before = roughness(noisy)
    # Interpolate noisy to AFTER_SCALE for fair comparison
    noisy_interp = model._interp(noisy, AFTER_SCALE)
    r_interp     = roughness(noisy_interp)
    r_after      = roughness(filtered_noisy)

    print(f"\n[Noisy sine]  roughness (raw interp)  = {r_interp:.4f}")
    print(f"              roughness (filter+interp)= {r_after:.4f}")
    print(f"  → {'SMOOTHER ✓' if r_after < r_interp else 'NOT smoother — check filter!'}")

    # ── 1c. Filter weight sanity ──
    w = model.low_pass.weight
    print(f"\n[Filter weights] shape={list(w.shape)}")
    print(f"  values (should be {1/KERNEL:.4f} at init): {w.squeeze().tolist()}")
    print(f"  sum per kernel (should be 1.0): {w.sum(dim=-1).squeeze().tolist()}")

    # ── Visualize ──
    fig, axes = plt.subplots(2, ACTION_DIM, figsize=(10, 6))
    time_in   = np.linspace(0, 1, CHUNK_SIZE)
    time_out  = np.linspace(0, 1, AFTER_SCALE)

    sample_noisy    = noisy[0].detach().numpy()
    sample_filtered = filtered_noisy[0].detach().numpy()
    sample_clean_interp = model._interp(sine_clean, AFTER_SCALE)[0].detach().numpy()

    for d in range(ACTION_DIM):
        ax = axes[0, d]
        ax.plot(time_in,   sample_noisy[:, d],    label="noisy input", alpha=0.6)
        ax.plot(time_out,  sample_clean_interp[:, d], label="clean interp", linestyle="--")
        ax.plot(time_out,  sample_filtered[:, d], label="filtered output", linewidth=2)
        ax.set_title(f"Dim {d}: input vs filtered")
        ax.legend(fontsize=7)

        ax2 = axes[1, d]
        ax2.plot(time_in, sample_noisy[:, d] - torch.sin(t_vals).numpy()
                 if d == 0 else sample_noisy[:, d] - torch.cos(t_vals).numpy(),
                 label="noise added", alpha=0.5)
        ax2.axhline(0, color="k", linewidth=0.5)
        ax2.set_title(f"Dim {d}: noise added to clean signal")
        ax2.legend(fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "filter_test.png")
    plt.savefig(out_path)
    print(f"\nVisualization saved → {out_path}")
    plt.close()


# ═══════════════════════════════════════════
# 2. LEARNING TEST
# ═══════════════════════════════════════════

def test_learning():
    print("\n" + "="*60)
    print("TEST 2: can the model learn?")
    print("="*60)

    torch.manual_seed(0)
    model = make_model(kernel_size=KERNEL)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ── Build a fixed synthetic dataset ──
    # State: random
    # Ground-truth action: smooth sine, same for all batches (easy to overfit)
    N_FIXED = 256
    t_vals  = torch.linspace(0, 2 * np.pi, CHUNK_SIZE)
    gt_action = torch.stack([torch.sin(t_vals), torch.cos(t_vals)], dim=-1)  # [CHUNK, DIM]
    gt_action = gt_action.unsqueeze(0).expand(N_FIXED, -1, -1)               # [N, CHUNK, DIM]
    states    = torch.randn(N_FIXED, STATE_DIM)

    # ── Print filter weights at init ──
    print(f"\nFilter weights at init: {model.low_pass.weight.squeeze().tolist()}")

    # ── Training loop ──
    NUM_STEPS = 1000
    losses = []
    for step in range(NUM_STEPS):
        idx = torch.randperm(N_FIXED)[:BATCH]
        s   = states[idx]
        a   = gt_action[idx]

        optim.zero_grad()
        loss = model.compute_loss(s, a)
        loss.backward()
        optim.step()
        losses.append(loss.item())

        if (step + 1) % 200 == 0:
            print(f"  step {step+1:4d}  loss={loss.item():.5f}")

    loss_drop = losses[0] / (losses[-1] + 1e-9)
    print(f"\nLoss at step 1   : {losses[0]:.5f}")
    print(f"Loss at step {NUM_STEPS}: {losses[-1]:.5f}")
    print(f"Loss reduction factor: {loss_drop:.1f}x  ({'OK ✓' if loss_drop > 2 else 'POOR — model not learning!'})")

    # ── Print filter weights after training ──
    print(f"\nFilter weights after training: {model.low_pass.weight.squeeze().tolist()}")
    w_sum = model.low_pass.weight.sum(dim=-1).squeeze()
    print(f"  kernel sum (should ideally stay near 1): {w_sum.tolist()}")

    # ── Inference: compare sample_actions vs gt_action (UNFILTERED) ──
    model.eval()
    with torch.no_grad():
        s_eval   = states[:BATCH]
        pred     = model.sample_actions(s_eval, num_steps=20)  # [BATCH, CHUNK, DIM]
        gt_eval  = gt_action[:BATCH]

        # MAE between prediction and ground truth (original, unfiltered)
        mae_unfiltered = (pred - gt_eval).abs().mean().item()

        # MAE between prediction and the FILTERED ground truth
        # (this is what the network was trained to predict in latent space)
        gt_filtered = model._filter(gt_eval)                          # [BATCH, AFTER_SCALE, DIM]
        gt_filtered_up = model._interp(gt_filtered, CHUNK_SIZE)       # upsample back for fair compare
        mae_filtered_up = (pred - gt_filtered_up).abs().mean().item()

    print(f"\nInference comparison (on training data, {BATCH} samples, 20 Euler steps):")
    print(f"  MAE vs original GT (unfiltered, upsampled): {mae_unfiltered:.4f}")
    print(f"  MAE vs filtered GT (filtered then upsampled): {mae_filtered_up:.4f}")
    print(f"  → {'Network is generating reasonable outputs ✓' if mae_unfiltered < 1.5 else 'Large error — check inference path!'}")

    # ── Visualize predictions ──
    fig, axes = plt.subplots(2, ACTION_DIM, figsize=(10, 6))
    t_full = np.linspace(0, 1, CHUNK_SIZE)

    pred_np      = pred[0].numpy()
    gt_np        = gt_eval[0].numpy()
    gt_filt_up_np= gt_filtered_up[0].numpy()

    for d in range(ACTION_DIM):
        ax = axes[0, d]
        ax.plot(t_full, gt_np[:, d],         label="GT (original)", linestyle="--", linewidth=2)
        ax.plot(t_full, gt_filt_up_np[:, d], label="GT (filtered→up)", linestyle=":", alpha=0.7)
        ax.plot(t_full, pred_np[:, d],        label="model output", linewidth=2)
        ax.set_title(f"Dim {d}: GT vs model output")
        ax.legend(fontsize=7)

        ax2 = axes[1, d]
        ax2.plot(range(NUM_STEPS), losses)
        ax2.set_xlabel("step")
        ax2.set_ylabel("loss")
        ax2.set_title("Training loss curve")
        ax2.set_yscale("log")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "learning_test.png")
    plt.savefig(out_path)
    print(f"Visualization saved → {out_path}")
    plt.close()


# ═══════════════════════════════════════════
# 3. POTENTIAL BUG CHECK: are gradients flowing through filter?
# ═══════════════════════════════════════════

def test_gradient_flow():
    print("\n" + "="*60)
    print("TEST 3: gradient flow through low_pass filter")
    print("="*60)

    model = make_model(kernel_size=KERNEL)

    s = torch.randn(BATCH, STATE_DIM)
    a = torch.randn(BATCH, CHUNK_SIZE, ACTION_DIM)

    loss = model.compute_loss(s, a)
    loss.backward()

    lp_grad = model.low_pass.weight.grad
    print(f"low_pass.weight.grad: {lp_grad.squeeze().tolist() if lp_grad is not None else 'None (no gradient!)'}")
    print(f"  → {'Gradient flows through filter ✓' if lp_grad is not None else 'NO gradient to filter — it is detached!'}")

    # Check if having a trainable filter could collapse weights to 0
    # (a sign of potential instability)
    print(f"\n  Note: filter weights ARE trainable (requires_grad={model.low_pass.weight.requires_grad})")
    print(f"  If the network compensates, the filter could drift and hurt smoothing.")
    print(f"  Consider: model.low_pass.weight.requires_grad_(False) to freeze it.")


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

if __name__ == "__main__":
    test_filter_correctness()
    test_learning()
    test_gradient_flow()
    print("\n" + "="*60)
    print("All tests done.")
    print("="*60)
