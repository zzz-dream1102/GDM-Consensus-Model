import numpy as np

class DecisionMaker:
    def __init__(self, loss_params):
        self.loss = loss_params

    def calculate_3wd_thresholds(self):
        # Eq. 20 (or 54): Risk thresholds alpha and beta
        l_bn, l_nn = self.loss['lambda_BN'], self.loss['lambda_NN']
        l_pb, l_bb = self.loss['lambda_PB'], self.loss['lambda_BB']
        l_pn = self.loss['lambda_PN']
        
        alpha_twd = (l_bn - l_nn) / ((l_bn - l_nn) + (l_pb - l_bb) + 1e-9)
        beta_twd = (l_bn - l_nn) / ((l_bn - l_nn) + (l_pn - l_bn) + 1e-9)
        return alpha_twd, beta_twd

    def calculate_probabilities(self, consensus_matrix, criteria_weights):
        # Eq. 21 (or 55): Pr(a_l) probability using static weights v_j
        return np.dot(criteria_weights, consensus_matrix)

    def topsis_ranking(self, consensus_matrix, criteria_weights, target_indices=None):
        # Eq. 22-23 (or 56): TOPSIS Ranking for closeness coefficient C_l*
        n, p = consensus_matrix.shape
        closeness_coeffs = -1 * np.ones(p) 
        if target_indices is None: target_indices = range(p)

        # z+ and z- ideal solutions
        z_plus = np.max(consensus_matrix, axis=1)
        z_minus = np.min(consensus_matrix, axis=1)
        
        for l in target_indices:
            u_l = consensus_matrix[:, l]
            # Eq. 22: Weighted Euclidean Distance D+ and D-
            d_plus = np.sqrt(np.sum(criteria_weights * (u_l - z_plus)**2))
            d_minus = np.sqrt(np.sum(criteria_weights * (u_l - z_minus)**2))
            
            # Eq. 23: Relative Closeness C_l*
            if d_plus + d_minus != 0:
                closeness_coeffs[l] = d_minus / (d_plus + d_minus)
            else:
                closeness_coeffs[l] = 0.0
        return closeness_coeffs