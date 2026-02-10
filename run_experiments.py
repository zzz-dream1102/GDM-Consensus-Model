import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# ---------------------------------------------------------
# 配置路径：确保能找到 src 文件夹下的模块
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

try:
    from hmsis_builder import HMSISBuilder
    # 注意：这里不需要 TrustCompletionModel，因为批量实验为了速度通常简化信任补全
    # 或者你可以导入它并像 main.py 那样使用，但这里我们模拟已补全的信任矩阵
    from weight_calculator import calculate_indegree_weight
    from consensus_measurer import ConsensusMeter
    from consensus_optimizer import ConsensusOptimizer
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保 'src' 文件夹存在，且包含所有模块文件。")
    sys.exit(1)

class ExperimentRunner:
    def __init__(self):
        self.base_config = {
            'n': 4, 'p': 5, 
            'max_iter': 30,
            'threshold': 0.85,
            'phi': 0.05
        }

    def run_single_simulation(self, m, alpha, gamma):
        """
        运行一次完整的仿真，返回 GCL 历史数据
        """
        # 1. 生成数据 (随机)
        raw_data = np.random.uniform(1, 9, size=(m, self.base_config['n'], self.base_config['p']))
        criteria_types = ['benefit'] * self.base_config['n']
        
        # HMSIS 处理
        hmsis = HMSISBuilder(criteria_types)
        u_scale1 = hmsis.calculate_scale1_utility(raw_data)
        
        # 2. 模拟信任矩阵
        # (为了批量实验的速度，这里直接生成一个具有一定密度的随机信任矩阵)
        trust_mask = (np.random.rand(m, m) < 0.3).astype(float)
        np.fill_diagonal(trust_mask, 1.0)
        trust_matrix = np.random.rand(m, m) * trust_mask
        np.fill_diagonal(trust_matrix, 1.0)
        
        # 3. 初始化 RL 优化器
        # 【关键】这里传入实验变量 alpha (对应代码里的 rho) 和 gamma
        optimizer = ConsensusOptimizer(
            consensus_threshold=self.base_config['threshold'],
            phi=self.base_config['phi'],
            rho=alpha,   # <--- 这里传入不同的学习率
            gamma=gamma,
            epsilon=0.1
        )
        
        meter = ConsensusMeter()
        current_opinions = u_scale1.copy()
        current_trust = trust_matrix.copy()
        
        history_gcl = []
        
        # 4. 迭代循环
        for t in range(self.base_config['max_iter']):
            weights = calculate_indegree_weight(current_trust)
            group_op = meter.aggregate_opinions(current_opinions, weights)
            cl_vals, gcl = meter.calculate_consensus_levels(current_opinions, group_op)
            cl_min = np.min(cl_vals)
            
            history_gcl.append(gcl)
            
            # 【修复点】只有当“没有冲突”（达成共识）时，才跳出循环
            if not meter.detect_conflicts(gcl, self.base_config['threshold']):
                break
                
            action_idx = optimizer.choose_action(gcl, cl_min)
            new_ops, new_trust, reward, cost, next_cl = optimizer.step(
                current_opinions, group_op, current_trust, cl_vals, gcl, action_idx
            )
            
            # RL 更新
            next_gcl = np.mean(next_cl)
            next_cl_min = np.min(next_cl)
            optimizer.update_q_table((gcl, cl_min), action_idx, reward, (next_gcl, next_cl_min))
            
            current_opinions = new_ops
            current_trust = new_trust
            
        return history_gcl

    def experiment_sensitivity_alpha(self):
        """
        实验 1: 学习率 Alpha 的敏感性分析
        对比 Alpha = 0.1, 0.5, 0.9 时的收敛曲线
        """
        print("\n[Experiment 1] Running Sensitivity Analysis on Alpha...")
        m = 10 
        alphas = [0.1, 0.5, 0.9]
        
        # 设置绘图风格
        plt.rcParams['axes.unicode_minus'] = False 
        plt.figure(figsize=(10, 6))
        
        for alpha in alphas:
            print(f"  Running for alpha = {alpha}...")
            # 为了平滑曲线，每个参数跑 10 次取平均
            avg_gcl = []
            max_len = 0
            for _ in range(10): 
                hist = self.run_single_simulation(m, alpha=alpha, gamma=0.9)
                if len(hist) > max_len: max_len = len(hist)
                avg_gcl.append(hist)
            
            # 数据对齐与平均
            plot_data = np.zeros(max_len)
            counts = np.zeros(max_len)
            for h in avg_gcl:
                for i, val in enumerate(h):
                    plot_data[i] += val
                    counts[i] += 1
            
            final_curve = []
            for i in range(max_len):
                if counts[i] > 0:
                    final_curve.append(plot_data[i] / counts[i])
            
            plt.plot(final_curve, label=f'$\\alpha={alpha}$', marker='.', markevery=2)

        plt.axhline(y=self.base_config['threshold'], color='r', linestyle='--', label='Threshold (0.85)')
        plt.title('Sensitivity Analysis of Learning Rate ($\\alpha$)')
        plt.xlabel('Iteration')
        plt.ylabel('Group Consensus Level (GCL)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
        print("实验 1 完成。")

    def experiment_large_scale(self):
        """
        实验 2: 大规模群体仿真 (Large-Scale GDM)
        对比 m = 10, 50, 100 时的收敛性能
        """
        print("\n[Experiment 2] Running Large-Scale Simulation...")
        group_sizes = [10, 50, 100]
        
        plt.figure(figsize=(10, 6))
        
        for m in group_sizes:
            print(f"  Simulating group size m = {m}...")
            # 跑 5 次取平均
            avg_gcl = []
            max_len = 0
            for _ in range(5): 
                hist = self.run_single_simulation(m, alpha=0.1, gamma=0.9)
                if len(hist) > max_len: max_len = len(hist)
                avg_gcl.append(hist)
            
            plot_data = np.zeros(max_len)
            counts = np.zeros(max_len)
            for h in avg_gcl:
                for i, val in enumerate(h):
                    plot_data[i] += val
                    counts[i] += 1
            
            final_curve = []
            for i in range(max_len):
                if counts[i] > 0:
                    final_curve.append(plot_data[i] / counts[i])
            
            plt.plot(final_curve, label=f'Group Size $m={m}$', linewidth=2)
            
        plt.axhline(y=self.base_config['threshold'], color='r', linestyle='--', label='Threshold (0.85)')
        plt.title('Convergence Performance under Different Group Sizes')
        plt.xlabel('Iteration')
        plt.ylabel('GCL')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
        print("实验 2 完成。")

if __name__ == "__main__":
    runner = ExperimentRunner()
    
    # 运行参数敏感性实验
    runner.experiment_sensitivity_alpha()
    
    # 运行大规模群体实验
    runner.experiment_large_scale()