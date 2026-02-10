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
        current_opinions = (raw_data - 1) / 8.0 # 简易归一化
        trust = np.random.rand(m, m)
        np.fill_diagonal(trust, 1.0)
        
        for t in range(train_max_steps):
            weights = calculate_indegree_weight(trust)
            group_op = meter.aggregate_opinions(current_opinions, weights)
            cl_vals, gcl = meter.calculate_consensus_levels(current_opinions, group_op)
            cl_min = np.min(cl_vals)
            
            if meter.detect_conflicts(gcl, optimizer.consensus_threshold):
                break
            
            action_idx = optimizer.choose_action(gcl, cl_min)
            new_ops, new_trust, reward, cost, next_cl = optimizer.step(
                current_opinions, group_op, trust, cl_vals, gcl, action_idx
            )
            
            # Update Q-table
            next_gcl = np.mean(next_cl)
            next_cl_min = np.min(next_cl)
            optimizer.update_q_table((gcl, cl_min), action_idx, reward, (next_gcl, next_cl_min))
            
            current_opinions = new_ops
            trust = new_trust
            
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
    FORCE_RETRAIN = False  # 如果设为 True，即使有存档也会强制重新训练
    
    # 0. 参数设置
    m, n, p = 10, 4, 5
    max_iterations = 20
    consensus_threshold = 0.85
    
    # 固定随机种子 (保证测试数据一致，不影响模型加载)
    np.random.seed(42) 
    torch.manual_seed(42)
    
    # 生成测试数据
    raw_data_test = np.random.uniform(1, 9, size=(m, n, p))
    criteria_types = ['benefit', 'benefit', 'cost', 'cost']
    criteria_weights = np.array([0.25, 0.25, 0.25, 0.25]) 
    initial_trust_mask = (np.random.rand(m, m) < 0.3).astype(np.float64)
    np.fill_diagonal(initial_trust_mask, 1.0)
    initial_trust_values = np.random.rand(m, m) * initial_trust_mask

    # Phase 1: HMSIS
    print("\n[Phase 1] 构建 HMSIS...")
    hmsis = HMSISBuilder(criteria_types, granular_size_G=3)
    u_scale1_test = hmsis.calculate_scale1_utility(raw_data_test)
    h_0_test = hmsis.process(raw_data_test)

    # ---------------------------------------------------------
    # Phase 2: GAT (带存档功能)
    # ---------------------------------------------------------
    print("\n[Phase 2] 信任网络模块...")
    input_dim = h_0_test.shape[1]
    trust_model = TrustCompletionModel(input_dim=input_dim, hidden_dim=64, output_embed_dim=32)
    
    gat_path = os.path.join(models_dir, 'gat_model.pth')
    
    if os.path.exists(gat_path) and not FORCE_RETRAIN:
        print(f"  -> 检测到已保存的模型: {gat_path}")
        print("  -> 正在加载...")
        trust_model.load_state_dict(torch.load(gat_path))
    else:
        print("  -> 未检测到模型或强制重训，开始训练...")
        trust_model = train_trust_model(
            trust_model, h_0_test, initial_trust_values, initial_trust_mask, epochs=200
        )
        print(f"  -> 保存模型至: {gat_path}")
        torch.save(trust_model.state_dict(), gat_path)
    
    # 推理
    trust_model.eval()
    with torch.no_grad():
        adj_tensor = torch.tensor(initial_trust_mask, dtype=torch.float32)
        predicted_trust_tensor, _ = trust_model(h_0_test, adj_tensor)
    completed_trust_test = predicted_trust_tensor.numpy()
    np.fill_diagonal(completed_trust_test, 1.0)
    print("  -> 信任矩阵准备就绪。")

    # ---------------------------------------------------------
    # Phase 5: RL (带存档功能)
    # ---------------------------------------------------------
    print("\n[Phase 5] 强化学习模块...")
    optimizer = ConsensusOptimizer(consensus_threshold=consensus_threshold, phi=0.05)
    
    q_table_path = os.path.join(models_dir, 'q_table.npy')
    
    if os.path.exists(q_table_path) and not FORCE_RETRAIN:
        print(f"  -> 检测到已保存的 Q-table: {q_table_path}")
        print("  -> 正在加载...")
        loaded_q = np.load(q_table_path)
        # 简单的形状检查
        if loaded_q.shape == optimizer.q_table.shape:
            optimizer.q_table = loaded_q
        else:
            print("  [警告] 保存的 Q-table 形状不匹配，重新训练。")
            optimizer = pretrain_rl_agent(optimizer, m, n, p, num_episodes=500)
            np.save(q_table_path, optimizer.q_table)
    else:
        print("  -> 未检测到 Q-table 或强制重训，开始预训练...")
        optimizer = pretrain_rl_agent(optimizer, m, n, p, num_episodes=500)
        print(f"  -> 保存 Q-table 至: {q_table_path}")
        np.save(q_table_path, optimizer.q_table)

    # ---------------------------------------------------------
    # Phase 3-5: 正式测试
    # ---------------------------------------------------------
    print("\n[Phase 3-5] 开始正式测试 (Inference Mode)...")
    optimizer.epsilon = 0.05 
    meter = ConsensusMeter()
    current_opinions = u_scale1_test.copy()
    current_trust = completed_trust_test.copy()
    
    history_gcl = []       
    history_cost = []      
    history_min_cl = [] 
    reached_consensus = False
    final_consensus_matrix = None
    
    for t in range(max_iterations):
        expert_weights = calculate_indegree_weight(current_trust)
        group_opinion = meter.aggregate_opinions(current_opinions, expert_weights)
        cl_values, gcl = meter.calculate_consensus_levels(current_opinions, group_opinion)
        cl_min = np.min(cl_values)
        
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
        new_ops, new_trust, reward, cost, next_cl = optimizer.step(
            current_opinions, group_opinion, current_trust, 
            cl_values, gcl, action_idx
        )
        
        # 在线学习并更新保存的 Q-table (可选)
        next_gcl = np.mean(next_cl)
        next_cl_min = np.min(next_cl)
        optimizer.update_q_table((gcl, cl_min), action_idx, reward, (next_gcl, next_cl_min))
        
        history_cost.append(cost)
        current_opinions = new_ops
        current_trust = new_trust
    
    # 每次跑完测试，也可以更新一下 Q-table 存档，越用越聪明
    # np.save(q_table_path, optimizer.q_table) 

    # ---------------------------------------------------------
    # 可视化 (保持不变)
    # ---------------------------------------------------------
    print("\n生成图表...")
    plt.rcParams['axes.unicode_minus'] = False 
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(len(history_gcl)), history_gcl, 'b-o', label='GCL', linewidth=2)
    plt.plot(range(len(history_min_cl)), history_min_cl, 'r--s', label='Min CL', alpha=0.6)
    plt.axhline(y=consensus_threshold, color='g', linestyle='--')
    plt.title('Consensus Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Level')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    ax1 = plt.gca()
    ax1.bar(range(len(history_cost)), history_cost, color='orange', alpha=0.5, label='Cost')
    ax1.set_ylabel('Cost')
    ax1.set_xlabel('Iteration')
    ax2 = ax1.twinx()
    ax2.plot(range(len(history_gcl)), history_gcl, 'b-x')
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