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
            'max_iter': 20, # 仿真通常不需要太长
            'threshold': 0.85,
            'phi': 0.05
        }
        # 即使是随机生成，也固定个种子方便复现，或者在循环里放开
        # np.random.seed(42) 

    def _train_gat_on_the_fly(self, h_0, initial_trust_values, mask, epochs=50):
        """
        [关键修正] 仿真内部的 GAT 现场训练
        为了保证逻辑闭环，每次随机生成数据后，都必须真的训练 GAT 来补全。
        为了速度，epochs 可以设小一点 (比如 50-100)，只求有补全效果。
        """
        input_dim = h_0.shape[1]
        # 动态实例化模型
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
            
        # 推理出补全后的矩阵
        model.eval()
        with torch.no_grad():
            completed_trust, _ = model(h_0_tensor, mask_tensor)
            
        T_completed = completed_trust.numpy()
        np.fill_diagonal(T_completed, 1.0)
        return T_completed

    def run_single_simulation(self, m, alpha, gamma):
        """
        运行一次“全链路”仿真
        流程: Random Data -> HMSIS -> GAT Training -> Completed Trust -> Consensus
        """
        # 1. 生成随机原始数据 (Raw Data)
        raw_data = np.random.uniform(1, 9, size=(m, self.base_config['n'], self.base_config['p']))
        criteria_types = ['benefit'] * self.base_config['n']
        
        # 2. Phase 1: HMSIS 处理
        hmsis = HMSISBuilder(criteria_types)
        u_scale1 = hmsis.calculate_scale1_utility(raw_data)
        h_0 = hmsis.process(raw_data) # 获取 GAT 输入特征
        
        # 3. 生成随机的“稀疏”信任网络 (Sparse Observation)
        # 这才是真正的“随机初始条件”，而不是预设结果
        # 先生成一个潜在的 Ground Truth (仅用于生成观测值)
        latent_trust = np.random.rand(m, m)
        np.fill_diagonal(latent_trust, 1.0)
        
        # 生成掩码 (Mask)，假设只有 30% 的边是已知的
        mask_ratio = 0.3
        trust_mask = (np.random.rand(m, m) < mask_ratio).astype(np.float64)
        np.fill_diagonal(trust_mask, 1.0)
        
        # 生成观测值 (输入给 GAT 的只有这部分)
        initial_trust_values = latent_trust * trust_mask
        
        # 4. Phase 2: GAT 补全 (Real Execution)
        # [核心] 这里调用 GAT 真的去算一遍补全，而不是直接用 latent_trust
        # 这样就模拟了“从有限信息恢复完整网络”的过程
        completed_trust = self._train_gat_on_the_fly(h_0, initial_trust_values, trust_mask, epochs=50)
        
        # 5. Phase 5: 强化学习共识 (RL Consensus)
        optimizer = ConsensusOptimizer(
            consensus_threshold=self.base_config['threshold'],
            phi=self.base_config['phi'],
            rho=alpha,   
            gamma=gamma,
            epsilon=0.1
        )
        
        meter = ConsensusMeter()
        current_opinions = u_scale1.copy()
        current_trust = completed_trust.copy() # 使用 GAT 算出来的矩阵
        
        history_gcl = []
        
        # 初始计算
        weights = calculate_indegree_weight(current_trust)
        group_op = meter.aggregate_opinions(current_opinions, weights)
        cl_vals, gcl = meter.calculate_consensus_levels(current_opinions, group_op)
        cl_min = np.min(cl_vals)
        
        # 迭代循环
        for t in range(self.base_config['max_iter']):
            history_gcl.append(gcl)
            
            if not meter.detect_conflicts(gcl, self.base_config['threshold']):
                break
                
            action_idx = optimizer.choose_action(gcl, cl_min)
            
            new_ops, new_trust, reward, cost, next_cl_vals = optimizer.step(
                current_opinions, group_op, current_trust, cl_vals, gcl, action_idx
            )
            
            # 更新
            next_gcl = np.mean(next_cl_vals)
            next_cl_min = np.min(next_cl_vals)
            optimizer.update_q_table((gcl, cl_min), action_idx, reward, (next_gcl, next_cl_min))
            
            current_opinions = new_ops
            current_trust = new_trust
            cl_vals = next_cl_vals
            gcl = next_gcl
            cl_min = next_cl_min
            
            weights = calculate_indegree_weight(current_trust)
            group_op = meter.aggregate_opinions(current_opinions, weights)
            
        return history_gcl

    def experiment_sensitivity_alpha(self):
        """
        实验 1: 学习率敏感性分析 (Monte Carlo Simulation)
        """
        print("\n[Experiment 1] Alpha 敏感性分析 (全链路仿真)...")
        m = 10 
        alphas = [0.1, 0.5, 0.9]
        
        plt.rcParams['axes.unicode_minus'] = False 
        plt.figure(figsize=(10, 6))
        
        for alpha in alphas:
            print(f"  正在仿真 alpha = {alpha} ...")
            # 跑 10 次取平均 (由于包含 GAT 训练，次数多会慢，10次通常够看趋势)
            avg_gcl = []
            max_len = 0
            n_simulations = 10 
            
            for i in range(n_simulations):
                if (i+1) % 2 == 0: print(f"    - Run {i+1}/{n_simulations}")
                hist = self.run_single_simulation(m, alpha=alpha, gamma=0.9)
                if len(hist) > max_len: max_len = len(hist)
                avg_gcl.append(hist)
            
            # 数据平均处理
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
        print("实验 1 完成。")

    def experiment_large_scale(self):
        """
        实验 2: 大规模群体仿真
        """
        print("\n[Experiment 2] 大规模群体仿真 (全链路)...")
        # 注意: 100个节点训练 GAT 会比较慢，演示时可适当调小或耐心等待
        group_sizes = [10, 30, 50] 
        
        plt.figure(figsize=(10, 6))
        
        for m in group_sizes:
            print(f"  仿真群体规模 m = {m} ...")
            avg_gcl = []
            max_len = 0
            n_simulations = 5 # 大规模跑5次取平均
            
            for i in range(n_simulations):
                print(f"    - Run {i+1}/{n_simulations}")
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
        print("实验 2 完成。")

if __name__ == "__main__":
    runner = ExperimentRunner()
    
    # 注意：全链路仿真速度较慢，因为包含了神经网络训练
    runner.experiment_sensitivity_alpha()
    runner.experiment_large_scale()