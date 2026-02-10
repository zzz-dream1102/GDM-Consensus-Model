import numpy as np

class ConsensusOptimizer:
    def __init__(self, consensus_threshold=0.85, phi=0.05, rho=0.1, gamma=0.9, epsilon=0.1):
        """
        初始化 RL 优化器 (完全严谨版)
        """
        self.consensus_threshold = consensus_threshold
        self.phi = phi
        self.lr = rho       # 学习率 alpha
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon
        
        # Q-table 初始化 (状态: 100, 动作: 5)
        self.n_states = 100
        self.n_actions = 5
        self.q_table = np.zeros((self.n_states, self.n_actions))
        
        # 动作空间: 调整力度 delta
        self.actions = [0.05, 0.10, 0.15, 0.20, 0.25]

        # 前景理论参数 (参考 Tversky & Kahneman 经典参数)
        self.pt_alpha = 0.88
        self.pt_beta = 0.88
        self.pt_lambda = 2.25

    def _get_state_index(self, gcl, cl_min):
        """状态离散化映射"""
        gcl_idx = int(gcl * 10)
        cl_min_idx = int(cl_min * 10)
        # 边界修正
        gcl_idx = min(max(gcl_idx, 0), 9)
        cl_min_idx = min(max(cl_min_idx, 0), 9)
        return gcl_idx * 10 + cl_min_idx

    def choose_action(self, gcl, cl_min):
        """Epsilon-Greedy 策略"""
        state_idx = self._get_state_index(gcl, cl_min)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state_idx])

    def _calculate_gcl_fast(self, opinions, weights):
        """
        [内部辅助函数] 快速计算当前的 GCL，用于 Reward 计算。
        为了保证严谨性，这里复现了 ConsensusMeter 的核心逻辑，
        避免了跨模块调用的复杂性，同时保证计算一致性。
        """
        # 1. 聚合群体意见
        # opinions: (m, n, p), weights: (m,)
        # 扩展权重维度以匹配意见矩阵: (m, 1, 1)
        w_expanded = weights[:, np.newaxis, np.newaxis]
        group_opinion = np.sum(opinions * w_expanded, axis=0) # (n, p)
        
        # 2. 计算距离 (Manhattan Distance / Total Variation)
        # abs(expert - group) -> sum over (n, p) -> divide by (n*p)
        dist = np.mean(np.abs(opinions - group_opinion), axis=(1, 2)) # (m,)
        
        # 3. 计算 CL (1 - dist)
        cl_values = 1 - dist
        
        # 4. 计算 GCL (加权平均)
        gcl = np.sum(cl_values * weights)
        
        return gcl, group_opinion, cl_values

    def prospect_value(self, delta_val):
        """前景理论价值函数 v(x)"""
        if delta_val >= 0:
            return delta_val ** self.pt_alpha
        else:
            return -self.pt_lambda * ((-delta_val) ** self.pt_beta)

    def step(self, current_opinions, old_group_op, trust_matrix, cl_values, old_gcl, action_idx):
        """
        执行动作，并计算【严谨的】前景理论奖励
        """
        delta = self.actions[action_idx]
        m, n, p = current_opinions.shape
        
        # --- 1. 执行意见修改 ---
        new_opinions = current_opinions.copy()
        
        # 计算权重 (用于后续重算 GCL)
        # 注意：这里我们假设本轮权重暂未因信任更新而剧烈变化，或者使用旧权重
        # 为了极度严谨，应该根据 new_trust 算 new_weights，但这里用 current_trust 近似是可以的
        # 只要计算 GCL 的逻辑是闭环的即可。
        # 这里我们需要计算入度权重，简单起见，我们假设权重在 step 内不突变，
        # 或者我们手动算一下权重（为了绝对严谨）：
        d_in = np.sum(trust_matrix, axis=0)
        weights = d_in / (np.sum(d_in) + 1e-9)
        
        raw_adjustment_sum = 0
        
        for i in range(m):
            if cl_values[i] < self.consensus_threshold:
                # 只有不达标的才改
                direction = old_group_op - current_opinions[i]
                change = delta * direction
                new_opinions[i] += change
                
                # 记录调整量 (作为 Cost)
                raw_adjustment_sum += np.sum(np.abs(change))
        
        new_opinions = np.clip(new_opinions, 0, 1)
        
        # --- 2. 信任演化 (Trust Evolution) ---
        new_trust = (1 - self.phi) * trust_matrix + self.phi * np.eye(m)
        np.fill_diagonal(new_trust, 1.0)
        
        # --- 3. 【核心修复】计算严谨的奖励 (Prospect Theory Reward) ---
        
        # A. 计算新的 GCL
        new_gcl, new_group_op, new_cl_vals = self._calculate_gcl_fast(new_opinions, weights)
        
        # B. 计算增量 (Gains/Losses)
        delta_gcl = new_gcl - old_gcl
        
        # C. 归一化成本 (Cost)
        # 将调整总量归一化到 0-1 之间，使其与 GCL 增量在同一量级
        norm_cost = raw_adjustment_sum / (m * n * p) 
        
        # D. 前景理论价值计算
        # 收益项：共识提升带来的快乐
        value_gain = self.prospect_value(delta_gcl)
        
        # 损失项：修改意见带来的痛苦 (Cost)
        # 注意：Cost 本身是正数，但在前景理论中是负效用
        # 我们把 -norm_cost 看作相对于“不修改”的损失
        value_loss = self.prospect_value(-norm_cost)
        
        # 总奖励
        reward = value_gain + value_loss
        
        # 额外激励：如果直接达成了共识，给予一个巨大的额外奖励
        if new_gcl >= self.consensus_threshold and old_gcl < self.consensus_threshold:
            reward += 1.0 
            
        return new_opinions, new_trust, reward, norm_cost, new_cl_vals

    def update_q_table(self, state_t, action_idx, reward, state_next):
        """标准 Q-learning 更新"""
        idx_t = self._get_state_index(state_t[0], state_t[1])
        idx_next = self._get_state_index(state_next[0], state_next[1])
        
        predict = self.q_table[idx_t, action_idx]
        target = reward + self.gamma * np.max(self.q_table[idx_next])
        
        self.q_table[idx_t, action_idx] += self.lr * (target - predict)