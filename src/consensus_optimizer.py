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
        """Epsilon-Greedy Action Selection"""
        state_idx = self._get_state_index(gcl, cl_min)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state_idx])

    def _calculate_gcl_strict(self, opinions, weights):
        """
        严格实现公式 (12)-(14): 共识度计算
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
        [STRICT FIX] 
        1. 解决因果律闭环: Trust -> Weight -> GCL -> TI -> Trust
        2. 严格符合 Eq. 10 (排除对角线自指信任)
        """
        theta = self.actions[action_idx] 
        m, n, p = current_opinions.shape
        
        # --- 1. Opinion Revision (Eq. 16) ---
        new_opinions = current_opinions.copy()
        cost_t = 0.0
        for i in range(m):
            if old_cl_values[i] < self.consensus_threshold:
                direction = old_group_op - current_opinions[i]
                adjustment = theta * direction
                new_opinions[i] += adjustment
                cost_t += theta
        new_opinions = np.clip(new_opinions, 0, 1)

        # --- 2. Fixed-Point Iteration for Trust & Weight Loop ---
        # 初始猜测：假设 Trust 暂时不变
        temp_trust = trust_matrix.copy()
        
        # 迭代求解直到收敛
        for _ in range(5):
            # A. 根据当前的 Trust 计算 Weights (Phase 3)
            # [CRITICAL FIX] Eq. 10: WT_i = sum_{k != i} t_{ki}
            # 必须在求和前将对角线置为 0
            trust_no_diag = temp_trust.copy()
            np.fill_diagonal(trust_no_diag, 0)
            
            d_in = np.sum(trust_no_diag, axis=0) # 按列求和
            
            # Eq. 11: Normalization
            denom = np.sum(d_in)
            if denom == 0: denom = 1e-9
            temp_weights = d_in / denom
            
            # B. 根据新 Weights 计算新的 Consensus Level (Phase 4)
            _, current_cl_iter = self._calculate_gcl_strict(new_opinions, temp_weights)
            
            # C. 计算 Delta CL 和激励 TI (Eq. 17)
            delta_cl = current_cl_iter - old_cl_values
            ti_values = np.array([self.calculate_trust_incentive(d) for d in delta_cl])
            
            # D. 更新 Trust (Eq. 18)
            next_trust_iter = trust_matrix.copy() # 必须基于原始 t 时刻矩阵更新
            for i in range(m):
                # 只有表现好的专家 i (ti > 0)，其被信任度 (第 i 列) 才会增加
                if ti_values[i] != 0:
                    next_trust_iter[:, i] = next_trust_iter[:, i] + self.phi * ti_values[i]
            
            # Eq. 18: Clip only (No forced diagonal reset)
            next_trust_iter = np.clip(next_trust_iter, 0, 1)
            
            # 更新状态进入下一次微迭代
            temp_trust = next_trust_iter

        # 循环结束，temp_trust 即为数学上满足所有方程的 T^(t+1)
        new_trust = temp_trust
        
        # 计算最终的 State_{t+1} (同样需要排除对角线)
        trust_final_no_diag = new_trust.copy()
        np.fill_diagonal(trust_final_no_diag, 0)
        d_in_final = np.sum(trust_final_no_diag, axis=0)
        denom_final = np.sum(d_in_final)
        if denom_final == 0: denom_final = 1e-9
        weights_final = d_in_final / denom_final
        
        final_gcl, final_cl_values = self._calculate_gcl_strict(new_opinions, weights_final)

        # --- 3. Reward Calculation (Eq. 19) ---
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