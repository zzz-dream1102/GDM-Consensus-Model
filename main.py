import numpy as np
import torch
import torch.optim as optim
import sys
import os
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 配置路径
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

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
    return model

# =========================================================
#  辅助函数：RL 预训练 (简单环境)
# =========================================================
def pretrain_rl_agent(optimizer, m, n, p, num_episodes=500):
    print(f"\n[Pre-training RL] 开始预训练 RL 智能体 ({num_episodes} Episodes)...")
    meter = ConsensusMeter()
    train_max_steps = 15 
    
    for episode in range(num_episodes):
        # 简化环境用于快速训练Q表
        raw_data = np.random.uniform(1, 9, size=(m, n, p))
        current_opinions = (raw_data - 1) / 8.0 
        trust = np.random.rand(m, m)
        np.fill_diagonal(trust, 1.0)
        
        weights = calculate_indegree_weight(trust)
        group_op = meter.aggregate_opinions(current_opinions, weights)
        cl_vals, gcl = meter.calculate_consensus_levels(current_opinions, group_op)
        cl_min = np.min(cl_vals)

        for t in range(train_max_steps):
            if meter.detect_conflicts(gcl, optimizer.consensus_threshold):
                break
            
            action_idx = optimizer.choose_action(gcl, cl_min)
            new_ops, new_trust, reward, cost, next_cl_vals = optimizer.step(
                current_opinions, group_op, trust, cl_vals, gcl, action_idx
            )
            
            next_gcl = np.mean(next_cl_vals)
            next_cl_min = np.min(next_cl_vals)
            
            optimizer.update_q_table((gcl, cl_min), action_idx, reward, (next_gcl, next_cl_min))
            
            current_opinions = new_ops
            trust = new_trust
            cl_vals = next_cl_vals
            gcl = next_gcl
            cl_min = next_cl_min
            
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
    
    FORCE_RETRAIN = False 
    m, n, p = 10, 4, 5
    max_iterations = 20
    consensus_threshold = 0.85
    
    np.random.seed(42) 
    torch.manual_seed(42)
    
    # 1. 生成 Ground Truth 数据 (保证实验逻辑真实性)
    print("\n[Phase 0] 数据生成...")
    raw_data_test = np.random.uniform(1, 9, size=(m, n, p))
    criteria_types = ['benefit', 'benefit', 'cost', 'cost']
    criteria_weights = np.array([0.25, 0.25, 0.25, 0.25]) 
    
    # 真实潜在信任网络
    true_trust_matrix = np.random.rand(m, m)
    np.fill_diagonal(true_trust_matrix, 1.0)
    
    # 观测掩码 (Mask)
    mask_ratio = 0.3
    initial_trust_mask = (np.random.rand(m, m) < mask_ratio).astype(np.float64)
    np.fill_diagonal(initial_trust_mask, 1.0)
    
    # GAT的输入只包含 Mask 部分
    initial_trust_values = true_trust_matrix * initial_trust_mask

    # 2. Phase 1: HMSIS
    print("\n[Phase 1] HMSIS 处理...")
    hmsis = HMSISBuilder(criteria_types, granular_size_G=3)
    u_scale1_test = hmsis.calculate_scale1_utility(raw_data_test)
    h_0_test = hmsis.process(raw_data_test)

    # 3. Phase 2: GAT
    print("\n[Phase 2] 信任补全...")
    input_dim = h_0_test.shape[1]
    trust_model = TrustCompletionModel(input_dim=input_dim, hidden_dim=64, output_embed_dim=32)
    
    gat_path = os.path.join(models_dir, 'gat_model.pth')
    
    if os.path.exists(gat_path) and not FORCE_RETRAIN:
        print(f"  -> 加载模型: {gat_path}")
        trust_model.load_state_dict(torch.load(gat_path))
    else:
        print("  -> 训练模型 (基于当前Mask数据)...")
        trust_model = train_trust_model(
            trust_model, h_0_test, initial_trust_values, initial_trust_mask, epochs=200
        )
        torch.save(trust_model.state_dict(), gat_path)
    
    # 推理
    trust_model.eval()
    with torch.no_grad():
        adj_tensor = torch.tensor(initial_trust_mask, dtype=torch.float32)
        predicted_trust_tensor, _ = trust_model(h_0_test, adj_tensor)
    
    completed_trust_test = predicted_trust_tensor.numpy()
    np.fill_diagonal(completed_trust_test, 1.0)

    # 4. Phase 5: RL Consensus
    print("\n[Phase 5] RL 共识达成...")
    optimizer = ConsensusOptimizer(consensus_threshold=consensus_threshold, phi=0.05)
    
    q_table_path = os.path.join(models_dir, 'q_table.npy')
    
    if os.path.exists(q_table_path) and not FORCE_RETRAIN:
        print(f"  -> 加载 Q-table: {q_table_path}")
        loaded_q = np.load(q_table_path)
        if loaded_q.shape == optimizer.q_table.shape:
            optimizer.q_table = loaded_q
        else:
            print("  [警告] 形状不匹配，重新训练。")
            optimizer = pretrain_rl_agent(optimizer, m, n, p, num_episodes=500)
    else:
        optimizer = pretrain_rl_agent(optimizer, m, n, p, num_episodes=500)
        np.save(q_table_path, optimizer.q_table)

    # 正式测试
    print("\n[Phase 3-5] 正式迭代测试...")
    optimizer.epsilon = 0.05 
    meter = ConsensusMeter()
    
    current_opinions = u_scale1_test.copy()
    current_trust = completed_trust_test.copy()
    
    history_gcl = []       
    history_cost = []      
    final_consensus_matrix = None
    
    weights = calculate_indegree_weight(current_trust)
    group_opinion = meter.aggregate_opinions(current_opinions, weights)
    cl_values, gcl = meter.calculate_consensus_levels(current_opinions, group_opinion)
    cl_min = np.min(cl_values)
    
    for t in range(max_iterations):
        history_gcl.append(gcl)
        print(f"  Test Iter {t}: GCL = {gcl:.4f} | Min CL = {cl_min:.4f}")
        
        if not meter.detect_conflicts(gcl, threshold=consensus_threshold):
            print(f"    >>> 达成共识! <<<")
            history_cost.append(0) 
            final_consensus_matrix = group_opinion
            break
            
        action_idx = optimizer.choose_action(gcl, cl_min)
        
        new_ops, new_trust, reward, cost, next_cl_vals = optimizer.step(
            current_opinions, group_opinion, current_trust, 
            cl_values, gcl, action_idx
        )
        
        next_gcl = np.mean(next_cl_vals)
        next_cl_min = np.min(next_cl_vals)
        optimizer.update_q_table((gcl, cl_min), action_idx, reward, (next_gcl, next_cl_min))
        
        history_cost.append(cost)
        
        current_opinions = new_ops
        current_trust = new_trust
        cl_values = next_cl_vals
        gcl = next_gcl
        cl_min = next_cl_min
        
        weights = calculate_indegree_weight(current_trust)
        group_opinion = meter.aggregate_opinions(current_opinions, weights)
    
    # 可视化部分
    print("\n生成图表...")
    plt.rcParams['axes.unicode_minus'] = False 
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    if history_gcl:
        plt.plot(range(len(history_gcl)), history_gcl, 'b-o', label='GCL', linewidth=2)
    plt.axhline(y=consensus_threshold, color='g', linestyle='--')
    plt.title('Consensus Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Level')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    if history_cost:
        plt.bar(range(len(history_cost)), history_cost, color='orange', alpha=0.5, label='Cost')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
    plt.title('Cost Trade-off')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # 5. Phase 6: Decision
    if final_consensus_matrix is not None:
        print("\n[Phase 6] 方案选择...")
        dm = DecisionMaker({'lambda_BN': 0.2, 'lambda_NN': 0.1, 'lambda_PB': 0.3, 'lambda_BB': 0.1, 'lambda_PN': 0.4})
        scores = dm.topsis_ranking(final_consensus_matrix, criteria_weights)
        
        valid_scores = [(idx, score) for idx, score in enumerate(scores)]
        valid_scores.sort(key=lambda x: x[1], reverse=True)
        print("最终排名:")
        for rank, (idx, score) in enumerate(valid_scores, 1):
             print(f"  Rank {rank}: A_{idx+1} ({score:.4f})")
    else:
        print("未达成共识。")

if __name__ == "__main__":
    main()