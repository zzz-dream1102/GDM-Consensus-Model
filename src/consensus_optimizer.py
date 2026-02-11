import numpy as np

class ConsensusOptimizer:
    def __init__(self, consensus_threshold=0.85, phi=0.05, rho=0.1, gamma=0.9, epsilon=0.1, 
                 delta_coeff=1.0, eta_coeff=1.0):
        """
        初始化 RL 优化器 (严格对应 formulas_final.txt)
        """
        self.consensus_threshold = consensus_threshold
        self.phi = phi          # Trust evolution rate (Eq. 18)
        self.lr = rho           # Learning rate (Eq. 19)
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.delta = delta_coeff
        self.eta = eta_coeff
        
        self.n_states = 100 
        self.actions = [0.05, 0.10, 0.15, 0.20, 0.25] 
        self.n_actions = len(self.actions)
        
        self.q_table = np.zeros((self.n_states, self.n_actions))

        self.pt_mu = 0.88      
        self.pt_nu = 0.88      
        self.pt_lambda = 2.25  

    def _get_state_index(self, gcl, cl_min):
        gcl_idx = int(gcl * 10)
        cl_min_idx = int(cl_min * 10)
        gcl_idx = min(max(gcl_idx, 0), 9)
        cl_min_idx = min(max(cl_min_idx, 0), 9)
        return gcl_idx * 10 + cl_min_idx

    def choose_action(self, gcl, cl_min):
        state_idx = self._get_state_index(gcl, cl_min)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state_idx])

    def _calculate_gcl_strict(self, opinions, weights):
        """
        辅助计算 GCL (对应 Eq. 12-14)
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
        """Eq. 17: PT Value Function"""
        if delta_cl >= 0:
            return delta_cl ** self.pt_mu  
        else:
            return -self.pt_lambda * ((-delta_cl) ** self.pt_nu)

    def step(self, current_opinions, old_group_op, trust_matrix, old_cl_values, old_gcl, action_idx):
        """
        执行 Phase 5 的单步迭代。
        [AUDIT FIX] 使用定点迭代法解决 Eq.17 和 Eq.12 之间的因果循环。
        """
        theta = self.actions[action_idx] 
        m, n, p = current_opinions.shape
        
        # --- 1. Opinion Revision (Eq. 50) ---
        new_opinions = current_opinions.copy()
        cost_t = 0.0
        for i in range(m):
            if old_cl_values[i] < self.consensus_threshold:
                direction = old_group_op - current_opinions[i]
                adjustment = theta * direction
                new_opinions[i] += adjustment
                cost_t += theta
        new_opinions = np.clip(new_opinions, 0, 1)

        # --- 2. Fixed-Point Iteration for Trust & Weights ---
        # 我们需要同时解出 t+1 时刻的 Trust, Weights, 和 CL
        # 初始猜测：假设 trust 暂时不变
        temp_trust = trust_matrix.copy()
        
        final_cl_values = None
        ti_values = None
        
        # 迭代求解平衡点 (通常 3-5 次即可收敛)
        for _ in range(5):
            # A. 基于当前的 temp_trust 计算权重 (Phase 3)
            d_in = np.sum(temp_trust, axis=0)
            temp_weights = d_in / (np.sum(d_in) + 1e-9)
            
            # B. 基于新权重计算 CL^(t+1) (Phase 4)
            _, current_cl_iter = self._calculate_gcl_strict(new_opinions, temp_weights)
            
            # C. 计算 Delta CL 和 TI (Eq. 17)
            delta_cl = current_cl_iter - old_cl_values
            ti_values = np.array([self.calculate_trust_incentive(d) for d in delta_cl])
            
            # D. 更新 Trust (Eq. 18)
            next_trust_iter = trust_matrix.copy()
            for i in range(m):
                if ti_values[i] != 0:
                    next_trust_iter[:, i] = next_trust_iter[:, i] + self.phi * ti_values[i]
            
            # [AUDIT FIX] 公式中只提到了 clip，没提到对角线重置，严格遵守公式则移除 fill_diagonal
            next_trust_iter = np.clip(next_trust_iter, 0, 1)
            
            # 检查收敛性 (可选，这里直接迭代固定次数)
            temp_trust = next_trust_iter
            final_cl_values = current_cl_iter

        # 循环结束，temp_trust 即为收敛后的 T^(t+1)
        new_trust = temp_trust
        
        # 计算最终状态 S_{t+1}
        d_in_final = np.sum(new_trust, axis=0)
        weights_final = d_in_final / (np.sum(d_in_final) + 1e-9)
        final_gcl, _ = self._calculate_gcl_strict(new_opinions, weights_final)

        # --- 3. Reward Calculation (Strict Eq. 19) ---
        # [AUDIT FIX] 移除了 +10.0 的虚构奖励
        r_pt = np.sum(ti_values)
        reward = self.delta * r_pt - self.eta * cost_t
            
        return new_opinions, new_trust, reward, cost_t, final_cl_values

    def update_q_table(self, state_t, action_idx, reward, state_next):
        """Eq. 19: Q-learning Update Rule"""
        idx_t = self._get_state_index(state_t[0], state_t[1])
        idx_next = self._get_state_index(state_next[0], state_next[1])
        
        predict = self.q_table[idx_t, action_idx]
        target = reward + self.gamma * np.max(self.q_table[idx_next])
        
        self.q_table[idx_t, action_idx] += self.lr * (target - predict)