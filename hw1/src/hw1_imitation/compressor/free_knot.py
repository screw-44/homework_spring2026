import torch
import numpy as np
import itertools

class FreeKnotBSpline:
    def __init__(self, seq_len, num_dof, num_cps, degree=3, sample_distance=1.0, joint_knot=True, device='cuda'):
        """
        枚举式 Free-Knot B-spline 动作压缩模块。
        
        Args:
            seq_len (int): 轨迹长度 T (如 8, 15)。
            num_dof (int): 动作维度 D。
            num_cps (int): 控制点数量 M (如 4, 6)。
            degree (int): B样条阶数，默认为 3 (Cubic B-spline)。
            sample_distance (float): 枚举自由节点时的步长，默认为1。
            joint_knot (bool): 是否所有的动作维度共享同一组自由节点。
            device (str): 计算设备。
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
        
        # 1. CPU 端预计算所有合法的节点组合 (空间换时间)
        if self.k > 0:
            # 生成候选节点网格 (不包含首尾点 0 和 T-1)
            grid = np.arange(1, self.seq_len - 1, self.sample_distance)
            # 使用组合(不允许重复且保持递增)来确保生成合法的样条节点
            combs = list(itertools.combinations(grid, self.k))
            self.knots_all = torch.tensor(combs, dtype=torch.float32, device=device) # [S, k]
        else:
            # k <= 0 代表退化为均匀边界样条 (如 15 -> 4), 没有自由节点
            self.knots_all = torch.empty((1, 0), dtype=torch.float32, device=device) # [1, 0]
            
        # 2. 生成所有组合对应的基矩阵 B_all 和伪逆 H_all
        self.S = self.knots_all.shape[0] # 组合总数 S
        self.B_all = self._get_batched_basis(self.knots_all) # [S, T, M]
        
        # 预计算所有基矩阵的伪逆，用于极速最小二乘拟合 [S, M, T]
        # (使用 pinv 以防某些极端的节点组合导致矩阵近似奇异)
        self.H_all = torch.linalg.pinv(self.B_all) 

    def _get_batched_basis(self, internal_knots):
        """
        Batched Cox-de Boor 算法，用于根据内部节点生成基函数矩阵。
        支持传入任意连续的 internal_knots，完美支持 decode 阶段。
        
        Args:
            internal_knots: [Batch, k]
        Returns:
            N: [Batch, T, num_cps] 基矩阵
        """
        B_eff = internal_knots.shape[0]
        
        # 构造 Clamped Knot Vector: 首尾各有 degree+1 个重复节点
        start_knots = torch.zeros((B_eff, self.degree + 1), device=internal_knots.device)
        end_knots = torch.full((B_eff, self.degree + 1), float(self.seq_len - 1), device=internal_knots.device)
        
        # full_knots: [B_eff, M + p + 1]
        full_knots = torch.cat([start_knots, internal_knots, end_knots], dim=1)
        num_knots = full_knots.shape[1]
        
        # 时间步 t:[B_eff, T]
        t = torch.arange(self.seq_len, dtype=torch.float32, device=internal_knots.device).unsqueeze(0).expand(B_eff, -1)
        
        # Cox-de Boor 递归 (d=0)
        N = torch.zeros(B_eff, self.seq_len, num_knots - 1, device=internal_knots.device)
        for i in range(num_knots - 1):
            left = full_knots[:, i:i+1]
            right = full_knots[:, i+1:i+2]
            mask = (t >= left) & (t < right)
            # 最后一个有效区间包含右边界
            if i == num_knots - self.degree - 2:
                mask = mask | (t == right)
            N[:, :, i] = mask.float()
            
        # Cox-de Boor 递归 (d=1 to degree)
        for d in range(1, self.degree + 1):
            N_new = torch.zeros(B_eff, self.seq_len, num_knots - 1 - d, device=internal_knots.device)
            for i in range(num_knots - 1 - d):
                left_num = t - full_knots[:, i:i+1]
                left_den = full_knots[:, i+d:i+d+1] - full_knots[:, i:i+1]
                left_term = torch.where(left_den > 1e-6, (left_num / left_den) * N[:, :, i], torch.zeros_like(N[:, :, i]))
                
                right_num = full_knots[:, i+d+1:i+d+2] - t
                right_den = full_knots[:, i+d+1:i+d+2] - full_knots[:, i+1:i+2]
                right_term = torch.where(right_den > 1e-6, (right_num / right_den) * N[:, :, i+1], torch.zeros_like(N[:, :, i+1]))
                
                N_new[:, :, i] = left_term + right_term
            N = N_new
            
        return N # [B_eff, T, M]

    def encode_continuous(self, trajs):
        """
        使用预计算的张量，通过 GPU 高度并行地枚举所有节点组合，选取 MSE 最小的一组。
        
        Args:
            trajs: [B, T, D] 真实轨迹
        Returns:
            latents:[B, num_params] 包含了 Control points 和 归一化的 Free-knots
        """
        B_size, T, D = trajs.shape
        
        # 1. 投影求出所有组合下的控制点 C: [B, S, M, D]
        # self.H_all:[S, M, T], trajs: [B, T, D]
        C = torch.einsum('smt, btd -> bsmd', self.H_all, trajs)
        
        # 2. 重构出所有组合下的轨迹 X_rec: [B, S, T, D]
        # self.B_all: [S, T, M], C: [B, S, M, D]
        X_rec = torch.einsum('stm, bsmd -> bstd', self.B_all, C)
        
        # 3. 计算重建误差并寻找最优解
        if self.joint_knot:
            # 所有自由度共用一套节点, MSE 维度是 [B, S]
            mse = torch.mean((X_rec - trajs.unsqueeze(1))**2, dim=(2, 3))
            best_idx = torch.argmin(mse, dim=1) # [B]
            
            # 提取最优控制点和最优节点
            C_best = C[torch.arange(B_size), best_idx, :, :] # [B, M, D]
            knots_best = self.knots_all[best_idx] # [B, k]
            
        else:
            # 不同的自由度各用一套节点, MSE 维度是 [B, S, D]
            mse = torch.mean((X_rec - trajs.unsqueeze(1))**2, dim=2)
            best_idx = torch.argmin(mse, dim=1) # [B, D]
            
            # 提取每个维度独有的最优控制点和最优节点
            C_best = torch.stack([C[torch.arange(B_size), best_idx[:, d], :, d] for d in range(D)], dim=-1) # [B, M, D]
            knots_best = torch.stack([self.knots_all[best_idx[:, d]] for d in range(D)], dim=1) #[B, D, k]
            
        # 4. Normalize 节点位置到 [0, 1] 范围
        if self.k > 0:
            knots_best = knots_best / (self.seq_len - 1.0)
            
        # 展平并拼接
        C_flat = C_best.reshape(B_size, -1) # [B, M * D]
        knots_flat = knots_best.reshape(B_size, -1) # [B, k] or[B, D * k]
        latents = torch.cat([C_flat, knots_flat], dim=-1) #[B, num_params]
        
        return latents

    def decode_continuous(self, latents):
        """
        从压缩的 latents 解码恢复出平滑的轨迹。由于这里输入的是连续值的 Free-knots，
        我们会动态调用 Batched Cox-de Boor 进行解码，完美支持下游预测微调。
        
        Args:
            latents: [B, num_params] 
        Returns:
            X_rec: [B, T, D] 重建轨迹
        """
        B_size = latents.shape[0]
        C_dim = self.num_cps * self.num_dof
        
        # 1. 拆包
        C_flat = latents[:, :C_dim]
        knots_flat = latents[:, C_dim:]
        C = C_flat.view(B_size, self.num_cps, self.num_dof) #[B, M, D]
        
        # 2. 还原 knots 并且保证其合法性
        if self.k > 0:
            knots = knots_flat * (self.seq_len - 1.0)
            knots = torch.clamp(knots, min=0.0, max=self.seq_len - 1.0)
            # 如果预测的 knots 逆序了，我们强行给它排序保持递增
            knots, _ = torch.sort(knots, dim=-1)
        else:
            # 没有自由节点
            if self.joint_knot:
                knots = torch.empty((B_size, 0), device=latents.device)
            else:
                knots = torch.empty((B_size * self.num_dof, 0), device=latents.device)

        # 3. 动态解码轨迹
        if self.joint_knot:
            # knots:[B, k]
            B_pred = self._get_batched_basis(knots) #[B, T, M]
            X_rec = torch.bmm(B_pred, C) #[B, T, M] x [B, M, D] -> [B, T, D]
        else:
            # knots: [B, D, k] -> [B*D, k]
            knots = knots.view(B_size * self.num_dof, self.k)
            B_pred = self._get_batched_basis(knots) # [B*D, T, M]
            
            # 变形 C: [B, M, D] -> [B, D, M] ->[B*D, M, 1]
            C_reshaped = C.transpose(1, 2).reshape(B_size * self.num_dof, self.num_cps, 1)
            X_rec_flat = torch.bmm(B_pred, C_reshaped) #[B*D, T, 1]
            X_rec = X_rec_flat.view(B_size, self.num_dof, self.seq_len).transpose(1, 2) #[B, T, D]
            
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