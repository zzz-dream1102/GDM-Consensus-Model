import numpy as np

class ConsensusOptimizer:
    def __init__(self, consensus_threshold=0.85, phi=0.05, rho=0.1, gamma=0.9, epsilon=0.1, 
                 delta_coeff=1.0, eta_coeff=1.0):
        """
        初始化 RL 优化器 (Strict Mathematical Compliance Version - Revised)
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
        
        # 动作空间: 对应公式中的调整步长 theta
        self.actions = [0.05, 0.10, 0.15, 0.20, 0.25] 

        # Prospect Theory Parameters 
        self.pt_mu = 0.88      
        self.pt_nu = 0.88      
        self.pt_lambda = 2.25  

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
        """
        计算前景理论信任激励 TI_i (Eq. 17)
        """
        if delta_cl >= 0:
            return delta_cl ** self.pt_mu  
        else:
            return -self.pt_lambda * ((-delta_cl) ** self.pt_nu)

    def step(self, current_opinions, old_group_op, trust_matrix, old_cl_values, old_gcl, action_idx):
        """
        执行 Phase 5 的单步迭代 (Revised for Causality)
        
        逻辑流:
        1. Action -> New Opinions
        2. New Opinions + OLD Weights -> Delta CL (衡量纯粹的意见改善) -> Trust Incentive
        3. Trust Incentive -> New Trust Matrix
        4. New Trust Matrix -> NEW Weights
        5. New Opinions + NEW Weights -> Final S_{t+1} (GCL, CL_min)
        """
        # 1. Action Selection
        theta = self.actions[action_idx] 
        m, n, p = current_opinions.shape
        
        new_opinions = current_opinions.copy()
        cost_t = 0.0
        
        # --- Opinion Revision (Eq. 16) ---
        for i in range(m):
            if old_cl_values[i] < self.consensus_threshold:
                direction = old_group_op - current_opinions[i]
                adjustment = theta * direction
                new_opinions[i] += adjustment
                # 只有实际调整了才计入成本
                cost_t += theta 
        
        new_opinions = np.clip(new_opinions, 0, 1)

        # --- Step 2: Calculate Trust Incentive (Based on OLD weights context) ---
        # 我们需要评估：在当前的群体结构(旧权重)下，你的意见改好了没有？
        # 计算旧权重的向量形式
        d_in_old = np.sum(trust_matrix, axis=0)
        weights_old = d_in_old / (np.sum(d_in_old) + 1e-9)
        
        # 使用新意见 + 旧权重 计算中间状态 CL'
        _, temp_cl_values = self._calculate_gcl_strict(new_opinions, weights_old)
        
        # Delta CL = CL'_{i} - CL_i^{(t)}
        delta_cl = temp_cl_values - old_cl_values
        
        # TI_i calculation
        ti_values = np.array([self.calculate_trust_incentive(d) for d in delta_cl])
        
        # --- Step 3: Trust Evolution (Eq. 18) -> T^{(t+1)} ---
        new_trust = trust_matrix.copy()
        for i in range(m):
            if ti_values[i] != 0:
                # 更新别人对 i 的信任 (第 i 列)
                new_trust[:, i] = new_trust[:, i] + self.phi * ti_values[i]
        
        new_trust = np.clip(new_trust, 0, 1)
        np.fill_diagonal(new_trust, 1.0) 
        
        # --- Step 4: Final State Calculation (Based on NEW weights) ---
        # 计算 t+1 时刻的新权重 W^{(t+1)}
        d_in_new = np.sum(new_trust, axis=0)
        weights_new = d_in_new / (np.sum(d_in_new) + 1e-9)
        
        # 计算最终的 GCL^{(t+1)} 和 CL_i^{(t+1)}
        # 这是 Agent 观测到的真实 Next State
        final_gcl, final_cl_values = self._calculate_gcl_strict(new_opinions, weights_new)
        
        # --- Reward Calculation ---
        r_pt = np.sum(ti_values)
        reward = self.delta * r_pt - self.eta * cost_t
        
        # 稀疏奖励 (基于最终达成的状态)
        if final_gcl >= self.consensus_threshold and old_gcl < self.consensus_threshold:
            reward += 10.0
            
        return new_opinions, new_trust, reward, cost_t, final_cl_values

    def update_q_table(self, state_t, action_idx, reward, state_next):
        """Eq. 19: Q-learning Update Rule"""
        idx_t = self._get_state_index(state_t[0], state_t[1])
        idx_next = self._get_state_index(state_next[0], state_next[1])
        
        predict = self.q_table[idx_t, action_idx]
        target = reward + self.gamma * np.max(self.q_table[idx_next])
        
        self.q_table[idx_t, action_idx] += self.lr * (target - predict)