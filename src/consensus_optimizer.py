import numpy as np

class ConsensusOptimizer:
    def __init__(self, consensus_threshold=0.85, phi=0.05, rho=0.1, gamma=0.9, epsilon=0.1, 
                 delta_coeff=1.0, eta_coeff=1.0):
        self.consensus_threshold = consensus_threshold
        self.phi = phi          # Eq. 18: Trust evolution rate phi
        self.lr = rho           # Eq. 19: Learning rate rho
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.delta = delta_coeff
        self.eta = eta_coeff
        
        self.n_states = 100 
        self.zeta_actions = [0.05, 0.10, 0.15, 0.20, 0.25] # Eq. 16: Action space for zeta
        self.n_actions = len(self.zeta_actions)
        self.q_table = np.zeros((self.n_states, self.n_actions))

        self.pt_mu = 0.88      
        self.pt_nu = 0.88      
        self.pt_lambda = 2.25  # Eq. 17: PT loss aversion coefficient

    def _get_state_index(self, gcl, cl_min):
        gcl_idx = min(max(int(gcl * 10), 0), 9)
        cl_min_idx = min(max(int(cl_min * 10), 0), 9)
        return gcl_idx * 10 + cl_min_idx

    def choose_action(self, gcl, cl_min):
        state_idx = self._get_state_index(gcl, cl_min)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state_idx])

    def _calculate_gcl_strict(self, opinions, omega_weights):
        # Eq. 12: Group Opinion Aggregation using expert weights omega
        w_expanded = omega_weights[:, np.newaxis, np.newaxis]
        group_opinion = np.sum(opinions * w_expanded, axis=0) 
        
        # Eq. 13: Individual Consensus Degree (CL_i)
        n, p = group_opinion.shape
        dist = np.sum(np.abs(opinions - group_opinion), axis=(1, 2)) / (n * p)
        cl_values = 1 - dist
        
        # Eq. 14: Group Consensus Level (GCL)
        gcl = np.mean(cl_values)
        return gcl, cl_values

    def calculate_trust_incentive(self, delta_cl):
        # Eq. 17: PT Value Function for Trust Incentive TI_i
        if delta_cl >= 0:
            return delta_cl ** self.pt_mu  
        else:
            return -self.pt_lambda * ((-delta_cl) ** self.pt_nu)

    def step(self, current_opinions, old_group_op, trust_matrix, old_cl_values, old_gcl, action_idx):
        zeta = self.zeta_actions[action_idx] # Eq. 16: Selection of zeta_i
        m, n, p = current_opinions.shape
        
        # --- 1. Opinion Revision (Eq. 16) ---
        new_opinions = current_opinions.copy()
        cost_t = 0.0
        for i in range(m):
            if old_cl_values[i] < self.consensus_threshold:
                direction = old_group_op - current_opinions[i]
                new_opinions[i] += zeta * direction
                cost_t += zeta
        new_opinions = np.clip(new_opinions, 0, 1)

        # --- 2. Fixed-Point Iteration for Trust & Weight Evolution ---
        temp_trust = trust_matrix.copy()
        for _ in range(5):
            # Phase 3: Update expert weights omega based on in-degree (Eq. 10-11)
            trust_no_diag = temp_trust.copy()
            np.fill_diagonal(trust_no_diag, 0)
            d_in = np.sum(trust_no_diag, axis=0) 
            denom = np.sum(d_in) if np.sum(d_in) != 0 else 1e-9
            omega_weights = d_in / denom
            
            # Phase 4: Calculate current CL
            _, current_cl_iter = self._calculate_gcl_strict(new_opinions, omega_weights)
            
            # Eq. 17: Calculate TI_i
            delta_cl = current_cl_iter - old_cl_values
            ti_values = np.array([self.calculate_trust_incentive(d) for d in delta_cl])
            
            # Eq. 18: Trust Matrix Evolution
            next_trust_iter = trust_matrix.copy()
            for i in range(m):
                if ti_values[i] != 0:
                    next_trust_iter[:, i] = np.clip(next_trust_iter[:, i] + self.phi * ti_values[i], 0, 1)
            temp_trust = next_trust_iter

        # Final state calculation for t+1
        new_trust = temp_trust
        trust_final_no_diag = new_trust.copy()
        np.fill_diagonal(trust_final_no_diag, 0)
        omega_final = np.sum(trust_final_no_diag, axis=0) / (np.sum(trust_final_no_diag) + 1e-9)
        final_gcl, final_cl_values = self._calculate_gcl_strict(new_opinions, omega_final)

        # Eq. 19: Reward Calculation
        r_pt = np.sum(ti_values)
        reward = self.delta * r_pt - self.eta * cost_t
            
        return new_opinions, new_trust, reward, cost_t, final_cl_values

    def update_q_table(self, state_t, action_idx, reward, state_next):
        # Eq. 19: Q-learning Update using rho (learning rate)
        idx_t = self._get_state_index(state_t[0], state_t[1])
        idx_next = self._get_state_index(state_next[0], state_next[1])
        predict = self.q_table[idx_t, action_idx]
        target = reward + self.gamma * np.max(self.q_table[idx_next])
        self.q_table[idx_t, action_idx] += self.lr * (target - predict)