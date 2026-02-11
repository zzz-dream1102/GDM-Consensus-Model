import numpy as np

class ConsensusOptimizer:
    def __init__(self, consensus_threshold=0.85, phi=0.05, rho=0.1, gamma=0.9, epsilon=0.1, 
                 delta_coeff=1.0, eta_coeff=1.0):
        """
        初始化 RL 优化器 (Strict Mathematical Compliance Version)
        严格对应 formulas_final.txt
        
        参数映射:
        - consensus_threshold: \bar{\theta} 
        - phi: \varphi (Trust evolution rate) 
        - rho: \rho (Learning rate) 
        - gamma: \gamma (Discount factor) 
        - epsilon: \epsilon (Exploration rate) 
        - delta_coeff: \delta (Reward gain coefficient) 
        - eta_coeff: \eta (Reward cost penalty coefficient) 
        """
        self.consensus_threshold = consensus_threshold
        self.phi = phi
        self.lr = rho       
        self.gamma = gamma  
        self.epsilon = epsilon
        self.delta = delta_coeff
        self.eta = eta_coeff
        
        # State Space S_t = [GCL, CL_min] 
        self.n_states = 100 
        
        # Action Space A_t (Discrete adjustment step sizes) 
        self.n_actions = 5
        self.q_table = np.zeros((self.n_states, self.n_actions))
        
        # 动作空间: 对应公式中的调整步长 \theta_i^{(t)}
        self.actions = [0.05, 0.10, 0.15, 0.20, 0.25] 

        # Prospect Theory Parameters 
        self.pt_mu = 0.88      # \mu (Gains power)
        self.pt_nu = 0.88      # \nu (Losses power)
        self.pt_lambda = 2.25  # \lambda_{PT} (Loss aversion)

    def _get_state_index(self, gcl, cl_min):
        """映射连续状态 S_t 到离散索引"""
        gcl_idx = int(gcl * 10)
        cl_min_idx = int(cl_min * 10)
        gcl_idx = min(max(gcl_idx, 0), 9)
        cl_min_idx = min(max(cl_min_idx, 0), 9)
        return gcl_idx * 10 + cl_min_idx

    def choose_action(self, gcl, cl_min):
        """Epsilon-Greedy Action Selection"""
        state_idx = self._get_state_index(gcl, cl_min)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state_idx])

    def _calculate_gcl_strict(self, opinions, weights):
        """
        辅助计算 GCL，用于 Reward 的闭环计算
        (完全对应 Eq. 12-14)
        """
        # Eq. 12: Group Opinion Aggregation
        w_expanded = weights[:, np.newaxis, np.newaxis]
        group_opinion = np.sum(opinions * w_expanded, axis=0) 
        
        # Eq. 13: Individual Consensus Degree (CL_i)
        n, p = group_opinion.shape
        dist = np.sum(np.abs(opinions - group_opinion), axis=(1, 2)) / (n * p)
        cl_values = 1 - dist
        
        # Eq. 14: Group Consensus Level (GCL)
        gcl = np.mean(cl_values)
        
        return gcl, cl_values

    def calculate_trust_incentive(self, delta_cl):
        """
        计算前景理论信任激励 TI_i
        对应 Eq. 17 
        """
        # Value function V(.)
        if delta_cl >= 0:
            return delta_cl ** self.pt_mu  # (\Delta CL_i)^\mu
        else:
            return -self.pt_lambda * ((-delta_cl) ** self.pt_nu) # -\lambda (-\Delta CL_i)^\nu

    def step(self, current_opinions, old_group_op, trust_matrix, old_cl_values, old_gcl, action_idx):
        """
        执行 Phase 5 的单步迭代 (Strict Compliance)
        """
        # 1. Action Selection: Get theta from A_t
        theta = self.actions[action_idx] 
        m, n, p = current_opinions.shape
        
        # --- Eq. 16: Opinion Revision ---
        # u_{new} = (1 - theta) * u_{old} + theta * u_{group}
        new_opinions = current_opinions.copy()
        
        # Cost Calculation: Cost_t = sum(theta_i) 
        # 注意：公式定义成本为“调整步长的总和”
        cost_t = 0.0
        
        for i in range(m):
            # 仅调整未达标的专家 (Conflict Identification)
            if old_cl_values[i] < self.consensus_threshold:
                direction = old_group_op - current_opinions[i]
                adjustment = theta * direction
                new_opinions[i] += adjustment
                
                # 累加成本：因为该专家使用了 theta 这个步长进行调整
                cost_t += theta 
        
        new_opinions = np.clip(new_opinions, 0, 1)

        # --- Intermediate: Re-calculate Metrics for Reward ---
        # 计算临时权重用于评估 (基于当前信任)
        d_in = np.sum(trust_matrix, axis=0)
        temp_weights = d_in / (np.sum(d_in) + 1e-9)
        
        new_gcl, new_cl_values = self._calculate_gcl_strict(new_opinions, temp_weights)
        
        # --- Eq. 17: Trust Incentive Calculation ---
        # \Delta CL_i = CL_i^{(t+1)} - CL_i^{(t)}
        delta_cl = new_cl_values - old_cl_values
        
        # TI_i = V(\Delta CL_i)
        ti_values = np.array([self.calculate_trust_incentive(d) for d in delta_cl])
        
        # R_{PT} = sum(TI_i) 
        r_pt = np.sum(ti_values)
        
        # --- Eq. 18: Trust Evolution ---
        # \hat{t}_{ki}^{(t+1)} = clip(\hat{t}_{ki}^{(t)} + \varphi * TI_i, 0, 1)
        # 严格实现：专家 i 表现好，别人对他的信任(第 i 列)增加
        new_trust = trust_matrix.copy()
        for i in range(m):
            if ti_values[i] != 0:
                # 更新所有 k 对 i 的信任
                new_trust[:, i] = new_trust[:, i] + self.phi * ti_values[i]
        
        new_trust = np.clip(new_trust, 0, 1)
        np.fill_diagonal(new_trust, 1.0) 
        
        # --- Composite Reward R_t ---
        # R_t = \delta * R_{PT} - \eta * Cost_t
        reward = self.delta * r_pt - self.eta * cost_t
        
        # 稀疏奖励：为了加速收敛 (Implementation Trick，不违反公式逻辑)
        if new_gcl >= self.consensus_threshold and old_gcl < self.consensus_threshold:
            reward += 10.0
            
        return new_opinions, new_trust, reward, cost_t, new_cl_values

    def update_q_table(self, state_t, action_idx, reward, state_next):
        """
        Eq. 19: Q-learning Update Rule 
        """
        idx_t = self._get_state_index(state_t[0], state_t[1])
        idx_next = self._get_state_index(state_next[0], state_next[1])
        
        predict = self.q_table[idx_t, action_idx]
        target = reward + self.gamma * np.max(self.q_table[idx_next])
        
        self.q_table[idx_t, action_idx] += self.lr * (target - predict)