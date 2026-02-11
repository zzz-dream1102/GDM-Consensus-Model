import numpy as np

class ConsensusOptimizer:
    def __init__(self, consensus_threshold=0.85, phi=0.05, rho=0.1, gamma=0.9, epsilon=0.1, 
                 delta_coeff=1.0, eta_coeff=1.0):
        """
        初始化 RL 优化器 (Strict Mathematical Compliance Version)
        
        参数对应公式符号:
        - consensus_threshold: \bar{\theta} [Eq. 35]
        - phi: \varphi (Trust evolution rate) [Eq. 18]
        - rho: \rho (Learning rate) [Eq. 19]
        - gamma: \gamma (Discount factor) [Eq. 19]
        - delta_coeff: \delta (Reward gain coefficient) [Eq. 30]
        - eta_coeff: \eta (Reward cost penalty coefficient) [Eq. 30]
        """
        self.consensus_threshold = consensus_threshold
        self.phi = phi
        self.lr = rho       
        self.gamma = gamma  
        self.epsilon = epsilon
        self.delta = delta_coeff
        self.eta = eta_coeff
        
        # Q-table 初始化 (状态空间 S_t, 动作空间 A_t)
        self.n_states = 100
        self.n_actions = 5
        self.q_table = np.zeros((self.n_states, self.n_actions))
        
        # 动作空间: adjustment step size \theta (Eq. 16)
        self.actions = [0.05, 0.10, 0.15, 0.20, 0.25]

        # 前景理论参数 (Eq. 17)
        self.pt_mu = 0.88      # \mu
        self.pt_nu = 0.88      # \nu
        self.pt_lambda = 2.25  # \lambda_{PT}

    def _get_state_index(self, gcl, cl_min):
        """Map continuous state S_t = [GCL, CL_min] to discrete index."""
        gcl_idx = int(gcl * 10)
        cl_min_idx = int(cl_min * 10)
        gcl_idx = min(max(gcl_idx, 0), 9)
        cl_min_idx = min(max(cl_min_idx, 0), 9)
        return gcl_idx * 10 + cl_min_idx

    def choose_action(self, gcl, cl_min):
        """Epsilon-Greedy Strategy."""
        state_idx = self._get_state_index(gcl, cl_min)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state_idx])

    def _calculate_gcl_strict(self, opinions, weights):
        """
        辅助计算 GCL，用于 Reward 计算。
        Strictly follows Eq. 12-14.
        """
        # Eq. 12: Group Opinion Aggregation
        w_expanded = weights[:, np.newaxis, np.newaxis]
        group_opinion = np.sum(opinions * w_expanded, axis=0) 
        
        # Eq. 13: Individual Consensus Degree (ICD / CL_i)
        n, p = group_opinion.shape
        dist = np.sum(np.abs(opinions - group_opinion), axis=(1, 2)) / (n * p)
        cl_values = 1 - dist
        
        # Eq. 14: Group Consensus Level (GCL)
        gcl = np.mean(cl_values) 
        
        return gcl, cl_values

    def calculate_trust_incentive(self, delta_cl):
        """
        计算信任激励 TI_i (Eq. 17)
        """
        # Prospect Theory Value Function V(.)
        if delta_cl >= 0:
            return delta_cl ** self.pt_mu
        else:
            return -self.pt_lambda * ((-delta_cl) ** self.pt_nu)

    def step(self, current_opinions, old_group_op, trust_matrix, old_cl_values, old_gcl, action_idx):
        """
        执行单步交互，严格对应 Phase 5 所有公式。
        """
        theta = self.actions[action_idx] # Action A_t determines step size \theta
        m, n, p = current_opinions.shape
        
        # --- 1. Opinion Revision (Eq. 16) ---
        new_opinions = current_opinions.copy()
        
        # 记录 Cost (Total adjustment)
        cost_t = 0.0
        
        for i in range(m):
            # 仅调整未达标的专家 (Conflict Identification)
            if old_cl_values[i] < self.consensus_threshold:
                # Eq. 16: u_new = (1-theta)u_old + theta * u_c
                # 等价于: u_new = u_old + theta * (u_c - u_old)
                diff = old_group_op - current_opinions[i]
                adjustment = theta * diff
                new_opinions[i] += adjustment
                
                # Accumulate modification cost (Sum of theta used)
                cost_t += theta 
        
        # 修正边界
        new_opinions = np.clip(new_opinions, 0, 1)

        # --- 2. Calculate New Metrics for Reward ---
        # 暂时使用旧权重计算新的 CL (因为权重更新在下一步，属于 t+1)
        d_in = np.sum(trust_matrix, axis=0)
        temp_weights = d_in / (np.sum(d_in) + 1e-9)
        
        new_gcl, new_cl_values = self._calculate_gcl_strict(new_opinions, temp_weights)
        
        # --- 3. Trust Incentive & Evolution (Eq. 17 & 18) ---
        # 计算共识增益 \Delta CL_i
        delta_cl = new_cl_values - old_cl_values
        
        # 计算信任激励 TI_i (Eq. 17)
        ti_values = np.array([self.calculate_trust_incentive(d) for d in delta_cl])
        
        # 计算 R_PT (Total Trust Reward)
        r_pt = np.sum(ti_values)
        
        # 信任演化 (Eq. 18): t_ki(new) = clip(t_ki + phi * TI_i)
        # 含义：专家 i 表现好，其他人 k 对 i 的信任增加
        new_trust = trust_matrix.copy()
        for i in range(m):
            # 更新第 i 列 (所有 k -> i 的信任)
            if ti_values[i] != 0:
                new_trust[:, i] = new_trust[:, i] + self.phi * ti_values[i]
        
        # 保持对角线为 1，并截断 [0, 1]
        new_trust = np.clip(new_trust, 0, 1)
        np.fill_diagonal(new_trust, 1.0)
        
        # --- 4. Composite Reward (Eq. 30 / Phase 5 Text) ---
        # R_t = delta * R_PT - eta * Cost_t
        reward = self.delta * r_pt - self.eta * cost_t
        
        # 额外：如果达成共识，给予大额奖励 (Sparse Reward，辅助收敛，不违背公式)
        if new_gcl >= self.consensus_threshold and old_gcl < self.consensus_threshold:
            reward += 10.0
            
        return new_opinions, new_trust, reward, cost_t, new_cl_values

    def update_q_table(self, state_t, action_idx, reward, state_next):
        """Eq. 19: Q-learning Update Rule"""
        idx_t = self._get_state_index(state_t[0], state_t[1])
        idx_next = self._get_state_index(state_next[0], state_next[1])
        
        predict = self.q_table[idx_t, action_idx]
        target = reward + self.gamma * np.max(self.q_table[idx_next])
        
        self.q_table[idx_t, action_idx] += self.lr * (target - predict)