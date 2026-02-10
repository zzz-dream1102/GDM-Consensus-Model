import numpy as np

def calculate_indegree_weight(trust_matrix):
    """
    实现 Phase 3: 基于信任网络的专家权重确定 (Model M1)。
    
    对应公式:
    - 公式 (10): 入度中心性计算 (In-degree Centrality)
    - 公式 (11): 权重归一化 (Weight Normalization)
    
    输入:
        trust_matrix: Numpy 数组，维度 (m, m)。代表补全后的信任矩阵 \tilde{T}。
                      行索引为发出信任的专家 k，列索引为接受信任的专家 i。
    
    输出:
        weights: Numpy 数组，维度 (m,)。代表归一化后的专家权重向量 W。
    """
    # 确保输入为 float 类型以避免计算误差
    T = np.array(trust_matrix, dtype=np.float64)
    m = T.shape[0]

    # --- 公式 (10): 计算入度中心性 WT_i ---
    # WT_i = sum_{k=1, k!=i}^m t_{ki}
    # 逻辑: 计算指向专家 i 的所有信任值之和，排除专家对自己的信任 (k != i)。
    
    # 步骤 1: 创建副本并将对角线元素置为 0
    # 这满足了公式中 k != i 的约束，排除自指信任
    T_no_diag = T.copy()
    np.fill_diagonal(T_no_diag, 0)
    
    # 步骤 2: 按列求和 (axis=0)
    # 对于每个列 i，计算所有行 k 的和
    wt_scores = np.sum(T_no_diag, axis=0) # 维度: (m,)

    # --- 公式 (11): 权重归一化 ---
    # W_i = WT_i / sum(WT)
    
    total_score = np.sum(wt_scores)
    
    # 防止除以零 (虽然在全连通信任网络中不太可能发生，但作为防御性编程)
    if total_score == 0:
        # 如果没有任何信任流，则分配均匀权重
        return np.ones(m, dtype=np.float64) / m
    
    weights = wt_scores / total_score # 维度: (m,)
    
    return weights