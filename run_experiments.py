import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os
import copy

# ---------------------------------------------------------
# 配置路径
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

try:
    from hmsis_builder import HMSISBuilder
    from trust_network import TrustCompletionModel
    from weight_calculator import calculate_indegree_weight
    from consensus_measurer import ConsensusMeter
    from consensus_optimizer import ConsensusOptimizer
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

class ExperimentRunner:
    def __init__(self):
        self.base_config = {
            'n': 4, 'p': 5, 
            'max_iter': 20, 
            'threshold': 0.85,
            'phi': 0.05
        }

    def _train_gat_on_the_fly(self, h_0, initial_trust_values, mask, epochs=60):
        """
        [关键函数] 仿真内部的 GAT 现场训练
        确保每次实验的信任补全都是基于当次数据的，而非预设结果。
        """
        input_dim = h_0.shape[1]
        model = TrustCompletionModel(input_dim=input_dim, hidden_dim=32, output_embed_dim=16)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        h_0_tensor = h_0.clone().detach()
        true_tensor = torch.tensor(initial_trust_values, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            pred, _ = model(h_0_tensor, mask_tensor)
            loss = model.compute_loss(pred, true_tensor, mask_tensor)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            completed_trust, _ = model(h_0_tensor, mask_tensor)
            
        T_completed = completed_trust.numpy()
        np.fill_diagonal(T_completed, 1.0)
        return T_completed

    def run_single_simulation(self, m, alpha, gamma):
        """
        全链路仿真: Random Data -> HMSIS -> GAT Training -> Completed Trust -> Consensus
        """
        # 1. 生成随机原始数据 (Raw Data)
        raw_data = np.random.uniform(1, 9, size=(m, self.base_config['n'], self.base_config['p']))
        criteria_types = ['benefit'] * self.base_config['n']
        
        # 2. Phase 1: HMSIS 处理
        hmsis = HMSISBuilder(criteria_types)
        u_scale1 = hmsis.calculate_scale1_utility(raw_data)
        h_0 = hmsis.process(raw_data) 
        
        # 3. 生成随机观测信任 (Sparse Observation)
        # 这是一个真实的“随机初始条件”
        latent_trust = np.random.rand(m, m) # 潜在真实值
        np.fill_diagonal(latent_trust, 1.0)
        
        # 掩码: 模拟只有 30% 关系已知
        mask_ratio = 0.3
        trust_mask = (np.random.rand(m, m) < mask_ratio).astype(np.float64)
        np.fill_diagonal(trust_mask, 1.0)
        
        initial_trust_values = latent_trust * trust_mask
        
        # 4. Phase 2: GAT 补全 (Real Execution)
        # 用 GAT 算出补全后的矩阵，而不是直接用 latent_trust
        completed_trust = self._train_gat_on_the_fly(h_0, initial_trust_values, trust_mask, epochs=60)
        
        # 5. Phase 5: 强化学习共识
        optimizer = ConsensusOptimizer(
            consensus_threshold=self.base_config['threshold'],
            phi=self.base_config['phi'],
            rho=alpha,   
            gamma=gamma,
            epsilon=0.1
        )
        
        meter = ConsensusMeter()
        current_opinions = u_scale1.copy()
        current_trust = completed_trust.copy()
        
        history_gcl = []
        
        # 初始计算
        weights = calculate_indegree_weight(current_trust)
        group_op = meter.aggregate_opinions(current_opinions, weights)
        cl_vals, gcl = meter.calculate_consensus_levels(current_opinions, group_op)
        cl_min = np.min(cl_vals)
        
        for t in range(self.base_config['max_iter']):
            history_gcl.append(gcl)
            
            if not meter.detect_conflicts(gcl, self.base_config['threshold']):
                break
                
            action_idx = optimizer.choose_action(gcl, cl_min)
            
            # 使用修正后的 step
            new_ops, new_trust, reward, cost, next_cl_vals = optimizer.step(
                current_opinions, group_op, current_trust, cl_vals, gcl, action_idx
            )
            
            # 更新 Q-table
            next_gcl = np.mean(next_cl_vals)
            next_cl_min = np.min(next_cl_vals)
            optimizer.update_q_table((gcl, cl_min), action_idx, reward, (next_gcl, next_cl_min))
            
            # 状态流转
            current_opinions = new_ops
            current_trust = new_trust
            cl_vals = next_cl_vals
            gcl = next_gcl
            cl_min = next_cl_min
            
            # 更新群体意见 (下一轮使用)
            weights = calculate_indegree_weight(current_trust)
            group_op = meter.aggregate_opinions(current_opinions, weights)
            
        return history_gcl

    def experiment_sensitivity_alpha(self):
        print("\n[Experiment 1] Alpha 敏感性分析 (全链路仿真)...")
        m = 10 
        alphas = [0.1, 0.5, 0.9]
        
        plt.rcParams['axes.unicode_minus'] = False 
        plt.figure(figsize=(10, 6))
        
        for alpha in alphas:
            print(f"  正在仿真 alpha = {alpha} ...")
            # 这里的次数 n_simulations 决定了实验的可信度
            # 由于包含了 GAT 训练，速度会比纯数值计算慢，建议设为 10-20
            avg_gcl = []
            max_len = 0
            n_simulations = 10 
            
            for i in range(n_simulations):
                hist = self.run_single_simulation(m, alpha=alpha, gamma=0.9)
                if len(hist) > max_len: max_len = len(hist)
                avg_gcl.append(hist)
            
            # 数据对齐与平均
            plot_data = np.zeros(max_len)
            counts = np.zeros(max_len)
            for h in avg_gcl:
                for idx, val in enumerate(h):
                    plot_data[idx] += val
                    counts[idx] += 1
            
            final_curve = []
            for idx in range(max_len):
                if counts[idx] > 0:
                    final_curve.append(plot_data[idx] / counts[idx])
            
            plt.plot(final_curve, label=f'$\\alpha={alpha}$', marker='.', markevery=2)

        plt.axhline(y=self.base_config['threshold'], color='r', linestyle='--', label='Threshold (0.85)')
        plt.title('Sensitivity Analysis of Learning Rate ($\\alpha$)')
        plt.xlabel('Iteration')
        plt.ylabel('Average GCL')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    def experiment_large_scale(self):
        print("\n[Experiment 2] 大规模群体仿真 (全链路)...")
        group_sizes = [10, 30, 50] 
        
        plt.figure(figsize=(10, 6))
        
        for m in group_sizes:
            print(f"  仿真群体规模 m = {m} ...")
            avg_gcl = []
            max_len = 0
            n_simulations = 5 
            
            for i in range(n_simulations):
                hist = self.run_single_simulation(m, alpha=0.1, gamma=0.9)
                if len(hist) > max_len: max_len = len(hist)
                avg_gcl.append(hist)
            
            plot_data = np.zeros(max_len)
            counts = np.zeros(max_len)
            for h in avg_gcl:
                for idx, val in enumerate(h):
                    plot_data[idx] += val
                    counts[idx] += 1
            
            final_curve = []
            for idx in range(max_len):
                if counts[idx] > 0:
                    final_curve.append(plot_data[idx] / counts[idx])
            
            plt.plot(final_curve, label=f'm={m}', linewidth=2)
            
        plt.axhline(y=self.base_config['threshold'], color='r', linestyle='--', label='Threshold (0.85)')
        plt.title('Convergence Performance under Different Group Sizes')
        plt.xlabel('Iteration')
        plt.ylabel('Average GCL')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.experiment_sensitivity_alpha()
    runner.experiment_large_scale()