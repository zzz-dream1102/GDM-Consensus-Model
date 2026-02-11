import numpy as np
import torch
import torch.optim as optim
import sys
import os
import matplotlib.pyplot as plt
import copy

# ---------------------------------------------------------
# 配置路径
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

# 创建保存模型的文件夹
models_dir = os.path.join(current_dir, 'saved_models')
os.makedirs(models_dir, exist_ok=True)

try:
    from hmsis_builder import HMSISBuilder
    from trust_network import TrustCompletionModel
    from weight_calculator import calculate_indegree_weight
    from consensus_measurer import ConsensusMeter
    from consensus_optimizer import ConsensusOptimizer
    from decision_support import DecisionMaker
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

# =========================================================
#  辅助函数：GAT 训练
# =========================================================
def train_trust_model(model, h_0, initial_trust_values, initial_trust_mask, epochs=200, lr=0.01):
    print(f"\n[Training GAT] 开始训练信任预测模型 ({epochs} Epochs)...")
    h_0_tensor = h_0.clone().detach()
    # 注意：这里的 target 应该是被 Mask 过的矩阵 (观测值)
    # 计算 Loss 时，Loss 函数内部会再次乘以 Mask，确保只计算已知边的误差
    true_matrix_tensor = torch.tensor(initial_trust_values, dtype=torch.float32)
    mask_tensor = torch.tensor(initial_trust_mask, dtype=torch.float32)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        predicted_matrix, _ = model(h_0_tensor, mask_tensor)
        loss = model.compute_loss(predicted_matrix, true_matrix_tensor, mask_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"  - Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")
    print("[Training GAT] 训练完成。")
    return model

# =========================================================
#  辅助函数：RL 预训练
# =========================================================
def pretrain_rl_agent(optimizer, m, n, p, num_episodes=500):
    print(f"\n[Pre-training RL] 开始预训练 RL 智能体 ({num_episodes} Episodes)...")
    meter = ConsensusMeter()
    train_max_steps = 15 
    
    for episode in range(num_episodes):
        # 简化环境生成用于快速训练
        raw_data = np.random.uniform(1, 9, size=(m, n, p))
        current_opinions = (raw_data - 1) / 8.0 
        trust = np.random.rand(m, m)
        np.fill_diagonal(trust, 1.0)
        
        # 预训练时需要计算初始 GCL 用于状态初始化
        weights = calculate_indegree_weight(trust)
        group_op = meter.aggregate_opinions(current_opinions, weights)
        cl_vals, gcl = meter.calculate_consensus_levels(current_opinions, group_op)
        cl_min = np.min(cl_vals)

        for t in range(train_max_steps):
            if meter.detect_conflicts(gcl, optimizer.consensus_threshold):
                # 已经达成共识则提前结束
                break
            
            action_idx = optimizer.choose_action(gcl, cl_min)
            
            # 使用新版 step 函数
            # 注意：step 函数内部已经处理了 weights 的重新计算和 Next State 的生成
            new_ops, new_trust, reward, cost, next_cl_vals = optimizer.step(
                current_opinions, group_op, trust, cl_vals, gcl, action_idx
            )
            
            # 外部只需根据 next_cl_vals 计算简单的统计量供下一次迭代使用
            next_gcl = np.mean(next_cl_vals)
            next_cl_min = np.min(next_cl_vals)
            
            optimizer.update_q_table((gcl, cl_min), action_idx, reward, (next_gcl, next_cl_min))
            
            # 状态更新
            current_opinions = new_ops
            trust = new_trust
            cl_vals = next_cl_vals
            gcl = next_gcl
            cl_min = next_cl_min
            
            # 这里的 group_op 需要在下一次循环前更新吗？
            # 是的，step 函数内部算了一次用于 Reward，但循环变量也需要更新
            weights_new = calculate_indegree_weight(trust)
            group_op = meter.aggregate_opinions(current_opinions, weights_new)
            
    print("[Pre-training RL] 预训练完成。")
    return optimizer

# =========================================================
#  主流程
# =========================================================
def main():
    print("==========================================")
    print("    GDM Algorithm Pipeline Simulation     ")
    print("==========================================")

    # --- 配置开关 ---
    FORCE_RETRAIN = False 
    
    # 0. 参数设置
    m, n, p = 10, 4, 5
    max_iterations = 20
    consensus_threshold = 0.85
    
    np.random.seed(42) 
    torch.manual_seed(42)
    
    # --- 修正后的数据生成逻辑 ---
    print("\n[Phase 0] 生成 Ground Truth 数据...")
    # 1. 生成原始评价矩阵
    raw_data_test = np.random.uniform(1, 9, size=(m, n, p))
    criteria_types = ['benefit', 'benefit', 'cost', 'cost']
    criteria_weights = np.array([0.25, 0.25, 0.25, 0.25]) 
    
    # 2. 生成真实的信任网络 (Full Ground Truth)
    # 先生成一个比较密集的真实网络，或者完全随机
    true_trust_matrix = np.random.rand(m, m)
    np.fill_diagonal(true_trust_matrix, 1.0)
    
    # 3. 生成观测掩码 (Observation Mask)
    # 模拟现实中我们只知道部分社交关系 (例如 30%)
    mask_ratio = 0.3
    initial_trust_mask = (np.random.rand(m, m) < mask_ratio).astype(np.float64)
    np.fill_diagonal(initial_trust_mask, 1.0) # 自己信任自己通常是已知的
    
    # 4. 生成观测到的信任矩阵 (用于 GAT 输入)
    # 未被 Mask 的地方设为 0 (或其他填充值，GAT 会处理)
    initial_trust_values = true_trust_matrix * initial_trust_mask

    # Phase 1: HMSIS
    print("\n[Phase 1] 构建 HMSIS...")
    hmsis = HMSISBuilder(criteria_types, granular_size_G=3)
    u_scale1_test = hmsis.calculate_scale1_utility(raw_data_test)
    h_0_test = hmsis.process(raw_data_test)

    # ---------------------------------------------------------
    # Phase 2: GAT
    # ---------------------------------------------------------
    print("\n[Phase 2] 信任网络模块...")
    input_dim = h_0_test.shape[1]
    trust_model = TrustCompletionModel(input_dim=input_dim, hidden_dim=64, output_embed_dim=32)
    
    gat_path = os.path.join(models_dir, 'gat_model.pth')
    
    if os.path.exists(gat_path) and not FORCE_RETRAIN:
        print(f"  -> 检测到已保存的模型: {gat_path}")
        trust_model.load_state_dict(torch.load(gat_path))
    else:
        print("  -> 开始训练 GAT...")
        # 传入观测数据进行训练
        trust_model = train_trust_model(
            trust_model, h_0_test, initial_trust_values, initial_trust_mask, epochs=200
        )
        torch.save(trust_model.state_dict(), gat_path)
    
    # 推理：补全信任矩阵
    trust_model.eval()
    with torch.no_grad():
        adj_tensor = torch.tensor(initial_trust_mask, dtype=torch.float32)
        predicted_trust_tensor, _ = trust_model(h_0_test, adj_tensor)
    
    completed_trust_test = predicted_trust_tensor.numpy()
    np.fill_diagonal(completed_trust_test, 1.0)
    
    # (可选) 可以在这里计算一下补全的准确率 MSE (对比 true_trust_matrix)
    mse = np.mean((completed_trust_test - true_trust_matrix)**2)
    print(f"  -> 信任矩阵补全完成。MSE vs Ground Truth: {mse:.4f}")

    # ---------------------------------------------------------
    # Phase 5: RL
    # ---------------------------------------------------------
    print("\n[Phase 5] 强化学习模块...")
    optimizer = ConsensusOptimizer(consensus_threshold=consensus_threshold, phi=0.05)
    
    q_table_path = os.path.join(models_dir, 'q_table.npy')
    
    if os.path.exists(q_table_path) and not FORCE_RETRAIN:
        print(f"  -> 检测到 Q-table: {q_table_path}")
        loaded_q = np.load(q_table_path)
        if loaded_q.shape == optimizer.q_table.shape:
            optimizer.q_table = loaded_q
        else:
            print("  [警告] 形状不匹配，重新训练。")
            optimizer = pretrain_rl_agent(optimizer, m, n, p, num_episodes=500)
    else:
        print("  -> 开始 RL 预训练...")
        optimizer = pretrain_rl_agent(optimizer, m, n, p, num_episodes=500)
        np.save(q_table_path, optimizer.q_table)

    # ---------------------------------------------------------
    # Phase 3-5: 正式测试
    # ---------------------------------------------------------
    print("\n[Phase 3-5] 开始正式测试 (Inference Mode)...")
    optimizer.epsilon = 0.05 
    meter = ConsensusMeter()
    
    current_opinions = u_scale1_test.copy()
    current_trust = completed_trust_test.copy() # 使用 GAT 补全后的矩阵作为初始 T
    
    history_gcl = []       
    history_cost = []      
    history_min_cl = [] 
    reached_consensus = False
    final_consensus_matrix = None
    
    # 初始状态计算
    weights = calculate_indegree_weight(current_trust)
    group_opinion = meter.aggregate_opinions(current_opinions, weights)
    cl_values, gcl = meter.calculate_consensus_levels(current_opinions, group_opinion)
    cl_min = np.min(cl_values)
    
    for t in range(max_iterations):
        history_gcl.append(gcl)
        history_min_cl.append(cl_min)
        
        print(f"  Test Iter {t}: GCL = {gcl:.4f} | Min CL = {cl_min:.4f}")
        
        if not meter.detect_conflicts(gcl, threshold=consensus_threshold):
            print(f"    >>> 达成共识! <<<")
            history_cost.append(0) 
            reached_consensus = True
            final_consensus_matrix = group_opinion
            break
            
        action_idx = optimizer.choose_action(gcl, cl_min)
        
        # 使用修正后的 step
        new_ops, new_trust, reward, cost, next_cl_vals = optimizer.step(
            current_opinions, group_opinion, current_trust, 
            cl_values, gcl, action_idx
        )
        
        # 统计 Next State
        next_gcl = np.mean(next_cl_vals)
        next_cl_min = np.min(next_cl_vals)
        
        optimizer.update_q_table((gcl, cl_min), action_idx, reward, (next_gcl, next_cl_min))
        
        history_cost.append(cost)
        
        # 更新环境变量
        current_opinions = new_ops
        current_trust = new_trust
        cl_values = next_cl_vals
        gcl = next_gcl
        cl_min = next_cl_min
        
        # 更新群体共识矩阵 (用于下一轮)
        weights_new = calculate_indegree_weight(current_trust)
        group_opinion = meter.aggregate_opinions(current_opinions, weights_new)
    
    # ---------------------------------------------------------
    # 可视化 (保持不变)
    # ---------------------------------------------------------
    print("\n生成图表...")
    plt.rcParams['axes.unicode_minus'] = False 
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    if history_gcl:
        plt.plot(range(len(history_gcl)), history_gcl, 'b-o', label='GCL', linewidth=2)
        plt.plot(range(len(history_min_cl)), history_min_cl, 'r--s', label='Min CL', alpha=0.6)
    plt.axhline(y=consensus_threshold, color='g', linestyle='--')
    plt.title('Consensus Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Level')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    if history_cost:
        ax1 = plt.gca()
        ax1.bar(range(len(history_cost)), history_cost, color='orange', alpha=0.5, label='Cost')
        ax1.set_ylabel('Cost')
        ax1.set_xlabel('Iteration')
        ax2 = ax1.twinx()
        # 对齐长度
        plot_len = min(len(history_gcl), len(history_cost))
        ax2.plot(range(plot_len), history_gcl[:plot_len], 'b-x')
        ax2.set_ylabel('GCL')
    plt.title('Cost Trade-off')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # Phase 6: 决策
    # ---------------------------------------------------------
    if final_consensus_matrix is not None:
        print("\n[Phase 6] 方案选择...")
        dm = DecisionMaker({'lambda_BN': 0.2, 'lambda_NN': 0.1, 'lambda_PB': 0.3, 'lambda_BB': 0.1, 'lambda_PN': 0.4})
        alpha, beta = dm.calculate_3wd_thresholds()
        probs = dm.calculate_probabilities(final_consensus_matrix, criteria_weights)
        cls_res = dm.classify_alternatives(probs, alpha, beta)
        pos_idx = [l for l, r in cls_res.items() if r == 'POS']
        if not pos_idx: pos_idx = list(range(p))
        scores = dm.topsis_ranking(final_consensus_matrix, criteria_weights, pos_idx)
        
        valid_scores = [(idx, score) for idx, score in enumerate(scores) if score >= 0]
        valid_scores.sort(key=lambda x: x[1], reverse=True)
        print("最终排名:")
        for rank, (idx, score) in enumerate(valid_scores, 1):
             print(f"  Rank {rank}: A_{idx+1} ({score:.4f})")
    else:
        print("未达成共识。")

if __name__ == "__main__":
    main()