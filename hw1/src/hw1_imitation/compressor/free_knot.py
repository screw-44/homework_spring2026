import torch
import numpy as np
import itertools

class FreeKnotBSpline:
    def __init__(self, seq_len, num_dof, num_cps, degree=3, sample_distance=1.0, joint_knot=True, device='cuda'):
        """
        基于枚举的 Free-Knot B-spline 动作压缩模块 (方案A: 极致精度预计算版)
        
        Args:
            seq_len (int): 轨迹长度 T
            num_dof (int): 动作维度 D
            num_cps (int): 控制点数量 M
            degree (int): B样条阶数，默认为 3
            sample_distance (float): 枚举自由节点时的步长，默认 1.0
            joint_knot (bool): 是否所有动作维度共享同一组自由节点
            device (str): 最终张量驻留和推理运行的设备
        """
        self.seq_len = seq_len
        self.num_dof = num_dof
        self.num_cps = num_cps
        self.degree = degree
        self.sample_distance = sample_distance
        self.joint_knot = joint_knot
        self.device = device
        
        # 内部自由节点数量 k = M - p - 1
        self.k = self.num_cps - self.degree - 1
        
        # 压缩向量总参数维度
        # C_flat: [M * D]，knots_flat: [k]（joint）或 [D * k]（non-joint）
        if self.joint_knot:
            self.num_param = self.num_cps * self.num_dof + self.k
        else:
            self.num_param = self.num_cps * self.num_dof + self.k * self.num_dof
        
        if self.k > 0:
            # 1. 在 CPU 上生成候选网格组合 (使用 float64 保证初始化算力精度)
            grid = np.arange(1, self.seq_len - 1, self.sample_distance)
            combs = list(itertools.combinations(grid, self.k))
            knots_cpu_double = torch.tensor(combs, dtype=torch.float64, device='cpu') # [S, k]
        else:
            knots_cpu_double = torch.empty((1, 0), dtype=torch.float64, device='cpu')
            
        self.S = knots_cpu_double.shape[0]
        
        # 2. 核心：在 CPU 上使用 Float64 计算极高精度的基矩阵和无偏伪逆
        # 这彻底消灭了病态矩阵在 Float32 下引发的数值截断和拟合崩塌问题
        B_all_double = self._get_batched_basis(knots_cpu_double) # [S, T, M]
        H_all_double = torch.linalg.pinv(B_all_double)           #[S, M, T]
        
        # 3. 精度降级：将完美的投影矩阵转回 Float32，并推送到 GPU 上供推理使用
        self.knots_all = knots_cpu_double.to(dtype=torch.float32, device=self.device)
        self.B_all = B_all_double.to(dtype=torch.float32, device=self.device)
        self.H_all = H_all_double.to(dtype=torch.float32, device=self.device)

    def _get_batched_basis(self, internal_knots):
        """
        Batched Cox-de Boor 算法。自适应 internal_knots 的 dtype。
        在 init 时运行在 CPU Float64，在 decode 时运行在 GPU Float32。
        """
        dtype = internal_knots.dtype
        device = internal_knots.device
        B_eff = internal_knots.shape[0]
        
        # 构造 Clamped Knot Vector: 首尾各有 degree+1 个重复节点
        start_knots = torch.zeros((B_eff, self.degree + 1), dtype=dtype, device=device)
        end_knots = torch.full((B_eff, self.degree + 1), float(self.seq_len - 1), dtype=dtype, device=device)
        full_knots = torch.cat([start_knots, internal_knots, end_knots], dim=1)
        num_knots = full_knots.shape[1]
        
        # 时间步 t:[B_eff, T]
        t = torch.arange(self.seq_len, dtype=dtype, device=device).unsqueeze(0).expand(B_eff, -1)
        
        N = torch.zeros(B_eff, self.seq_len, num_knots - 1, dtype=dtype, device=device)
        for i in range(num_knots - 1):
            left = full_knots[:, i:i+1]
            right = full_knots[:, i+1:i+2]
            mask = (t >= left) & (t < right)
            if i == num_knots - self.degree - 2:
                mask = mask | (t == right)
            N[:, :, i] = mask.to(dtype)
            
        for d in range(1, self.degree + 1):
            N_new = torch.zeros(B_eff, self.seq_len, num_knots - 1 - d, dtype=dtype, device=device)
            for i in range(num_knots - 1 - d):
                left_num = t - full_knots[:, i:i+1]
                left_den = full_knots[:, i+d:i+d+1] - full_knots[:, i:i+1]
                
                # 安全除法：防止 Float32 推理时底层抛出 0/0=NaN 污染显存
                left_den_safe = torch.where(left_den > 1e-6, left_den, torch.ones_like(left_den))
                left_term = torch.where(left_den > 1e-6, (left_num / left_den_safe) * N[:, :, i], torch.zeros_like(N[:, :, i]))
                
                right_num = full_knots[:, i+d+1:i+d+2] - t
                right_den = full_knots[:, i+d+1:i+d+2] - full_knots[:, i+1:i+2]
                
                right_den_safe = torch.where(right_den > 1e-6, right_den, torch.ones_like(right_den))
                right_term = torch.where(right_den > 1e-6, (right_num / right_den_safe) * N[:, :, i+1], torch.zeros_like(N[:, :, i+1]))
                
                N_new[:, :, i] = left_term + right_term
            N = N_new
            
        return N

    def encode_continuous(self, trajs):
        """
        使用 GPU 高度并行地枚举所有节点组合，返回最优的控制点和 Normalized Knots。
        Args: trajs [Batch, T, D]
        Returns: latents[Batch, num_params]
        """
        B_size, T, D = trajs.shape
        
        # 1. 直接进行纯张量运算投影 (零分支流，极速)
        C = torch.einsum('smt, btd -> bsmd', self.H_all, trajs)     #[B, S, M, D]
        X_rec = torch.einsum('stm, bsmd -> bstd', self.B_all, C)    # [B, S, T, D]
        
        # 2. 依据是否 Joint Knots 计算 MSE 并检索最优组合
        if self.joint_knot:
            mse = torch.mean((X_rec - trajs.unsqueeze(1))**2, dim=(2, 3)) # [B, S]
            best_idx = torch.argmin(mse, dim=1) # [B]
            
            C_best = C[torch.arange(B_size), best_idx, :, :] # [B, M, D]
            knots_best = self.knots_all[best_idx]            # [B, k]
        else:
            mse = torch.mean((X_rec - trajs.unsqueeze(1))**2, dim=2)      # [B, S, D]
            best_idx = torch.argmin(mse, dim=1)                           # [B, D]
            
            C_best = torch.stack([C[torch.arange(B_size), best_idx[:, d], :, d] for d in range(D)], dim=-1) # [B, M, D]
            knots_best = torch.stack([self.knots_all[best_idx[:, d]] for d in range(D)], dim=1)             #[B, D, k]
            
        # 3. 对节点位置进行 Normalize，使其除以轨迹长度范围映射至 [0, 1] 方便网络建模
        if self.k > 0:
            knots_best = knots_best / (self.seq_len - 1.0)
            
        # 4. 展平组合
        C_flat = C_best.reshape(B_size, -1)
        knots_flat = knots_best.reshape(B_size, -1)
        latents = torch.cat([C_flat, knots_flat], dim=-1)
        
        return latents

    def decode_continuous(self, latents):
        """
        从压缩向量重构出轨迹。支持通过网络微调预测连续值 knots。
        Args: latents [Batch, num_params]
        Returns: X_rec [Batch, T, D]
        """
        B_size = latents.shape[0]
        C_dim = self.num_cps * self.num_dof
        
        # 1. 拆分提取特征
        C_flat = latents[:, :C_dim]
        knots_flat = latents[:, C_dim:]
        C = C_flat.view(B_size, self.num_cps, self.num_dof) # [B, M, D]
        
        # 2. 还原 knots 到真实时间刻度并进行合法性约束 (强制排序防越界)
        if self.k > 0:
            if self.joint_knot:
                # joint 情况：knots_flat 形状为 [B, k]，直接 denorm + 排序
                knots = knots_flat * (self.seq_len - 1.0)                    # [B, k]
                knots = torch.clamp(knots, min=0.0, max=self.seq_len - 1.0)
                knots, _ = torch.sort(knots, dim=-1)                         # [B, k]
            else:
                # ★ Bug 修复：non-joint 时 knots_flat 形状为 [B, D*k]
                # 必须先 reshape 至 [B, D, k]，再对每个维度的 k 个节点独立排序
                # 若直接对 [B, D*k] 做 sort(dim=-1) 会把所有维度的节点混在一起排序，导致错位！
                knots = knots_flat.view(B_size, self.num_dof, self.k) * (self.seq_len - 1.0)  # [B, D, k]
                knots = torch.clamp(knots, min=0.0, max=self.seq_len - 1.0)
                knots, _ = torch.sort(knots, dim=-1)                         # [B, D, k] 每维度独立排序
                knots = knots.view(B_size * self.num_dof, self.k)            # [B*D, k]
        else:
            knots_shape = (B_size, 0) if self.joint_knot else (B_size * self.num_dof, 0)
            knots = torch.empty(knots_shape, dtype=latents.dtype, device=latents.device)

        # 3. 动态生成预测的基矩阵并解码重构
        if self.joint_knot:
            B_pred = self._get_batched_basis(knots) #[B, T, M]
            X_rec = torch.bmm(B_pred, C)            # [B, T, D]
        else:
            # knots 已经是 [B*D, k]，直接生成基矩阵
            B_pred = self._get_batched_basis(knots) #[B*D, T, M]
            
            # 使用 contiguous 保证显存块连续对齐后再进行维度重组
            C_reshaped = C.transpose(1, 2).contiguous().view(B_size * self.num_dof, self.num_cps, 1)
            X_rec_flat = torch.bmm(B_pred, C_reshaped) # [B*D, T, 1]
            
            X_rec = X_rec_flat.contiguous().view(B_size, self.num_dof, self.seq_len).transpose(1, 2) # [B, T, D]
            
        return X_rec
    

# ======== 测试用例 ========
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    B, T, D = 5, 16, 5  # 5 synthetic trajectories: linear, sin, cos, quadratic, random
    M = 6  # Number of control points
    
    # Build compression module
    spline = FreeKnotBSpline(seq_len=T, num_dof=D, num_cps=M, joint_knot=False)
    print(f"Search space size (enumeration combinations S): {spline.S}") 
    
    # Generate 5 synthetic trajectories
    t_norm = torch.linspace(0, 1, T).cuda()
    t_actual = torch.linspace(0, T-1, T).cuda()  # Actual time axis
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
    trajs[:, :, 4] = torch.randn(B, T).cuda()
    
    # Replicate trajectory across batch dimension
    for b in range(B):
        trajs[b] = trajs[0]
    
    print(f"\nOriginal trajectory shape: {trajs.shape}")
    print(f"Trajectory dimension description:")
    print(f"  - Dim 0: Linear (2.5*t + 1.0)")
    print(f"  - Dim 1: sin(2π*t)")
    print(f"  - Dim 2: cos(2π*t)")
    print(f"  - Dim 3: t²")
    print(f"  - Dim 4: random noise")
    
    # Encode
    latents = spline.encode_continuous(trajs)
    print(f"\nCompressed latents shape: {latents.shape}")
    
    # Decode
    reconstructed = spline.decode_continuous(latents)
    print(f"Reconstructed trajectory shape: {reconstructed.shape}")
    
    # Calculate MSE error
    mse = torch.mean((trajs - reconstructed) ** 2, dim=1)  # [B, D]
    print(f"\nMSE error (by dimension):")
    dim_names = ["linear", "sin", "cos", "quadratic", "random"]
    for d in range(D):
        mse_d = mse[:, d].mean().item()
        print(f"  Dim {d} ({dim_names[d]}): {mse_d:.6f}")
    print(f"Overall MSE: {torch.mean(mse).item():.6f}")
    
    # Visualization (use first batch)
    fig, axes = plt.subplots(D, 1, figsize=(12, 4*D))
    if D == 1:
        axes = [axes]
    
    for d in range(D):
        ax = axes[d]
        
        # Original and reconstructed trajectories
        original = trajs[0, :, d].cpu().numpy()
        recon = reconstructed[0, :, d].cpu().numpy()
        t_plot = t_actual.cpu().numpy()  # Use actual time axis
        
        ax.plot(t_plot, original, 'o-', label='Original trajectory', linewidth=2, markersize=6)
        ax.plot(t_plot, recon, 's--', label='Reconstructed trajectory', linewidth=2, markersize=5, alpha=0.7)
        
        # Visualize knot positions
        if spline.joint_knot:
            # Get optimal knots for this dimension
            best_idx = torch.argmin(torch.mean((reconstructed[0:1] - trajs[0:1]) ** 2, dim=2)[:, d])
            knots = spline.knots_all[best_idx] * (T - 1.0)
        else:
            # Different dimensions have different knots
            C_dim = spline.num_cps * spline.num_dof
            knots_flat = latents[0, C_dim:].view(spline.num_dof, -1)
            knots = knots_flat[d] * (T - 1.0)
        
        # Mark knot positions with vertical lines
        for knot_val in knots:
            ax.axvline(knot_val.item(), color='red', linestyle=':', alpha=0.5, linewidth=1.5)
        
        mse_d = mse[0, d].item()
        ax.set_title(f'Dim {d} ({dim_names[d]}) - MSE: {mse_d:.6f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Action value')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/hxy/Desktop/homework_spring2026/hw1/free_knot_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: /home/hxy/Desktop/homework_spring2026/hw1/free_knot_visualization.png")
    plt.show()