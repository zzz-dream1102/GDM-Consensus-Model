import numpy as np

class DecisionMaker:
    """
    实现 Phase 6: 基于三支决策 (3WD) 和 TOPSIS 的方案选择。
    Strictly complies with Eq. 53 - Eq. 56 in formulas_final.txt.
    """
    def __init__(self, loss_params):
        """
        初始化决策支持模块。
        """
        self.loss = loss_params

    def calculate_3wd_thresholds(self):
        """
        实现公式 (54): 计算三支决策的概率阈值 alpha 和 beta。
        """
        l_bn = self.loss['lambda_BN']
        l_nn = self.loss['lambda_NN']
        l_pb = self.loss['lambda_PB']
        l_bb = self.loss['lambda_BB']
        l_pn = self.loss['lambda_PN']
        
        # alpha = (L_BN - L_NN) / [ (L_BN - L_NN) + (L_PB - L_BB) ]
        numerator_alpha = l_bn - l_nn
        denominator_alpha = (l_bn - l_nn) + (l_pb - l_bb)
        
        # beta = (L_BN - L_NN) / [ (L_BN - L_NN) + (L_PN - L_BN) ]
        numerator_beta = l_bn - l_nn
        denominator_beta = (l_bn - l_nn) + (l_pn - l_bn)
        
        if denominator_alpha == 0: denominator_alpha = 1e-9
        if denominator_beta == 0: denominator_beta = 1e-9
            
        alpha = numerator_alpha / denominator_alpha
        beta = numerator_beta / denominator_beta
        
        return alpha, beta

    def calculate_probabilities(self, consensus_matrix, criteria_weights):
        """
        实现公式 (55): 计算每个方案的综合概率 Pr(A_l)。
        Pr(A_l) = sum_j (v_j * u_{c,j,l})
        """
        # 矩阵乘法: (n,) dot (n, p) -> (p,)
        probabilities = np.dot(criteria_weights, consensus_matrix)
        return probabilities

    def classify_alternatives(self, probabilities, alpha, beta):
        """
        执行三支决策分类 (Partitioning)。
        """
        m = probabilities.shape[0]
        classification = {}
        
        for l in range(m):
            prob = probabilities[l]
            if prob >= alpha:
                classification[l] = 'POS' # Positive Region (Accept)
            elif prob <= beta:
                classification[l] = 'NEG' # Negative Region (Reject)
            else:
                classification[l] = 'BND' # Boundary Region (Defer)
                
        return classification

    def topsis_ranking(self, consensus_matrix, criteria_weights, target_indices=None):
        """
        Strict implementation of Eq. 56 for TOPSIS Ranking.
        
        Input:
            consensus_matrix: (n, p) Numpy array. u_{c,j,l}^*
            criteria_weights: (n,) Numpy array. v_j
        Output:
            closeness_coeffs: (p,) Numpy array.
        """
        n, p = consensus_matrix.shape
        closeness_coeffs = -1 * np.ones(p) 
        
        if target_indices is None:
            target_indices = range(p)
        if len(target_indices) == 0:
            return closeness_coeffs

        # 1. 确定 PIS (Z+) 和 NIS (Z-)
        # 这里的理想解是基于原始共识矩阵的，而不是加权矩阵
        # z_j^+ = max_l {u_{c,j,l}}
        z_plus = np.max(consensus_matrix, axis=1) # (n,)
        z_minus = np.min(consensus_matrix, axis=1) # (n,)
        
        # 2. 计算距离 D+ 和 D- (严格对应 Eq. 56)
        # D_l = sqrt( sum_j v_j * (u - z)^2 )
        # 注意：v_j 是线性的，不是平方的
        
        for l in target_indices:
            u_l = consensus_matrix[:, l] # (n,)
            
            # Distance to PIS
            diff_sq_plus = (u_l - z_plus) ** 2
            # 权重直接乘在平方项外
            d_plus = np.sqrt(np.sum(criteria_weights * diff_sq_plus))
            
            # Distance to NIS
            diff_sq_minus = (u_l - z_minus) ** 2
            d_minus = np.sqrt(np.sum(criteria_weights * diff_sq_minus))
            
            # 3. 计算相对贴近度 (Eq. 56 下半部分)
            # C_l = D- / (D+ + D-)
            if d_plus + d_minus == 0:
                score = 0.0
            else:
                score = d_minus / (d_plus + d_minus)
            
            closeness_coeffs[l] = score
            
        return closeness_coeffs