"""
Compare Free-Knot B-Spline vs BEAST (uniform knots) compression.
This test verifies that free-knot implementation achieves lower MSE than uniform knots.
"""

import torch
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/hxy/Desktop/homework_spring2026/hw1/src')

from hw1_imitation.compressor.free_knot import FreeKnotBSpline
from hw1_imitation.compressor.beast import BeastTokenizer


def test_free_knot_vs_beast():
    # Test configuration
    B, T, D = 5, 16, 5  # 5 synthetic trajectories: linear, sin, cos, quadratic, random
    M = 6  # Number of control points
    
    # Generate synthetic trajectories (same as in free_knot.py)
    t_norm = torch.linspace(0, 1, T).cuda()
    t_actual = torch.linspace(0, T-1, T).cuda()
    trajs = torch.zeros(B, T, D).cuda()
    
    # Dim 0: Linear function ax + b (a=2.5, b=1.0)
    trajs[:, :, 0] = 2.5 * t_norm + 1.0
    
    # Dim 1: sin curve
    trajs[:, :, 1] = torch.sin(2 * torch.pi * t_norm)
    
    # Dim 2: cos curve
    trajs[:, :, 2] = torch.cos(2 * torch.pi * t_norm)
    
    # Dim 3: quadratic curve (t^2)
    trajs[:, :, 3] = t_norm ** 2
    
    # Dim 4: random noise
    torch.manual_seed(42)  # For reproducibility
    trajs[:, :, 4] = torch.randn(B, T).cuda()
    
    # Replicate trajectory across batch dimension
    for b in range(B):
        trajs[b] = trajs[0]
    
    print(f"Test trajectory shape: {trajs.shape}")
    print(f"Trajectory dimension description:")
    print(f"  - Dim 0: Linear (2.5*t + 1.0)")
    print(f"  - Dim 1: sin(2π*t)")
    print(f"  - Dim 2: cos(2π*t)")
    print(f"  - Dim 3: t²")
    print(f"  - Dim 4: random noise")
    print()
    
    # ============= Method 1: Free-Knot B-Spline =============
    print("=" * 60)
    print("Method 1: Free-Knot B-Spline (Non-uniform knots)")
    print("=" * 60)
    
    spline_free = FreeKnotBSpline(seq_len=T, num_dof=D, num_cps=M, joint_knot=False)
    print(f"Search space size (enumeration combinations): {spline_free.S}")
    
    # Encode and decode with free knots
    latents_free = spline_free.encode_continuous(trajs)
    reconstructed_free = spline_free.decode_continuous(latents_free)
    
    # Calculate MSE
    mse_free = torch.mean((trajs - reconstructed_free) ** 2, dim=1)  # [B, D]
    print(f"\nFree-Knot MSE (by dimension):")
    dim_names = ["linear", "sin", "cos", "quadratic", "random"]
    for d in range(D):
        mse_d = mse_free[:, d].mean().item()
        print(f"  Dim {d} ({dim_names[d]}): {mse_d:.6f}")
    overall_mse_free = torch.mean(mse_free).item()
    print(f"Overall MSE: {overall_mse_free:.6f}")
    print()
    
    # ============= Method 2: BEAST (Uniform knots) =============
    print("=" * 60)
    print("Method 2: BEAST (Uniform knots)")
    print("=" * 60)
    
    # Initialize BEAST tokenizer using the local implementation
    beast = BeastTokenizer(
        num_dof=D,
        num_basis=M,
        seq_len=T,
        degree_p=3,
        device='cuda'
    )
    
    # BEAST supports batch dimension
    # Encode: [B, num_basis * D]
    latents_beast = beast.encode_continuous(trajs, update_bounds=True)
    
    # Decode: [B, T, D]
    reconstructed_beast = beast.decode_continuous(latents_beast)
    
    # Calculate MSE
    mse_beast = torch.mean((trajs - reconstructed_beast) ** 2, dim=1)  # [B, D]
    print(f"\nBEAST MSE (by dimension):")
    for d in range(D):
        mse_d = mse_beast[:, d].mean().item()
        print(f"  Dim {d} ({dim_names[d]}): {mse_d:.6f}")
    overall_mse_beast = torch.mean(mse_beast).item()
    print(f"Overall MSE: {overall_mse_beast:.6f}")
    print()
    
    # ============= Comparison =============
    print("=" * 60)
    print("Comparison Results")
    print("=" * 60)
    print(f"\nOverall MSE:")
    print(f"  Free-Knot: {overall_mse_free:.6f}")
    print(f"  BEAST:     {overall_mse_beast:.6f}")
    print(f"  Improvement: {((overall_mse_beast - overall_mse_free) / overall_mse_beast * 100):.2f}%")
    
    print(f"\nPer-dimension comparison:")
    for d in range(D):
        mse_free_d = mse_free[:, d].mean().item()
        mse_beast_d = mse_beast[:, d].mean().item()
        improvement = ((mse_beast_d - mse_free_d) / mse_beast_d * 100) if mse_beast_d > 0 else 0
        better = "✓" if mse_free_d < mse_beast_d else "✗"
        print(f"  {dim_names[d]:10s}: Free={mse_free_d:.6f}, BEAST={mse_beast_d:.6f}, " +
              f"Improvement={improvement:6.2f}% {better}")
    
    # Verification
    all_better = all(mse_free[:, d].mean() < mse_beast[:, d].mean() for d in range(D))
    print(f"\n{'✓' if all_better else '✗'} Free-Knot achieves lower MSE for all dimensions: {all_better}")
    print()
    
    # ============= Visualization =============
    print("Creating visualization...")
    fig, axes = plt.subplots(D, 1, figsize=(14, 4*D))
    if D == 1:
        axes = [axes]
    
    for d in range(D):
        ax = axes[d]
        
        # Original, free-knot, and BEAST trajectories
        original = trajs[0, :, d].cpu().numpy()
        recon_free = reconstructed_free[0, :, d].cpu().numpy()
        recon_beast = reconstructed_beast[0, :, d].cpu().numpy()
        t_plot = t_actual.cpu().numpy()
        
        ax.plot(t_plot, original, 'o-', label='Original', linewidth=2, markersize=8, color='blue')
        ax.plot(t_plot, recon_free, 's--', label='Free-Knot', linewidth=2, markersize=6, alpha=0.8, color='green')
        ax.plot(t_plot, recon_beast, '^:', label='BEAST (uniform)', linewidth=2, markersize=6, alpha=0.8, color='orange')
        
        # Visualize free-knot positions
        C_dim = spline_free.num_cps * spline_free.num_dof
        if spline_free.joint_knot:
            # joint_knot=True: 所有维度共享同一组节点 [k]，对每个 dim 画相同的竖线
            knots = latents_free[0, C_dim:] * (T - 1.0)  # [k]
        else:
            # joint_knot=False: 每个维度有独立节点 [D, k]
            knots_flat = latents_free[0, C_dim:].view(spline_free.num_dof, -1)
            knots = knots_flat[d] * (T - 1.0)  # [k]
        
        for knot_val in knots:
            ax.axvline(knot_val.item(), color='green', linestyle=':', alpha=0.3, linewidth=1.5, label='Free-knot positions' if knot_val == knots[0] else '')
        
        # MSE values
        mse_free_d = mse_free[0, d].item()
        mse_beast_d = mse_beast[0, d].item()
        improvement = ((mse_beast_d - mse_free_d) / mse_beast_d * 100) if mse_beast_d > 0 else 0
        
        ax.set_title(f'Dim {d} ({dim_names[d]}) - Free-Knot MSE: {mse_free_d:.6f}, BEAST MSE: {mse_beast_d:.6f} (↓{improvement:.1f}%)',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Action value')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = '/home/hxy/Desktop/homework_spring2026/hw1/test/comparison_free_knot_vs_beast.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.show()
    
    return overall_mse_free < overall_mse_beast


if __name__ == "__main__":
    success = test_free_knot_vs_beast()
    
    print("=" * 60)
    if success:
        print("✓ TEST PASSED: Free-Knot achieves lower overall MSE than BEAST")
        print("  This verifies that the free-knot implementation is correct.")
    else:
        print("✗ TEST FAILED: BEAST achieves lower MSE than Free-Knot")
        print("  This suggests an issue with the free-knot implementation.")
    print("=" * 60)
