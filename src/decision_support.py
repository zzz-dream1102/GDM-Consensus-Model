import numpy as np

class DecisionMaker:
    """
    实现 Phase 6: 基于三支决策 (3WD) 和 TOPSIS 的方案选择。
    Alternative Selection via 3WD and TOPSIS.
    """
    def __init__(self, loss_params):
        """
        初始化决策支持模块。
        
        Args:
            loss_params (dict): 包含风险损失参数的字典。
                必须包含 keys: 'lambda_BN', 'lambda_NN', 'lambda_PB', 'lambda_BB', 'lambda_PN'
                通常: 
                - P: Positive (Accept)
                - B: Boundary (Defer)
                - N: Negative (Reject)
                例如 lambda_BN 代表: 本该属于 N 但被决策为 B 的损失。
        """
        self.loss = loss_params

    def calculate_3wd_thresholds(self):
        """
        实现公式 (20): 计算三支决策的概率阈值 alpha 和 beta。
        
        Output:
            alpha (float): 接受/边界的阈值。
            beta (float): 边界/拒绝的阈值。
        """
        l_bn = self.loss['lambda_BN']
        l_nn = self.loss['lambda_NN']
        l_pb = self.loss['lambda_PB']
        l_bb = self.loss['lambda_BB']
        l_pn = self.loss['lambda_PN']
        
        # --- 公式 (20) ---
        # alpha = (L_BN - L_NN) / [ (L_BN - L_NN) + (L_PB - L_BB) ]
        numerator_alpha = l_bn - l_nn
        denominator_alpha = (l_bn - l_nn) + (l_pb - l_bb)
        
        # beta = (L_BN - L_NN) / [ (L_BN - L_NN) + (L_PN - L_BN) ]
        numerator_beta = l_bn - l_nn
        denominator_beta = (l_bn - l_nn) + (l_pn - l_bn)
        
        # 避免除以零
        if denominator_alpha == 0: denominator_alpha = 1e-9
        if denominator_beta == 0: denominator_beta = 1e-9
            
        alpha = numerator_alpha / denominator_alpha
        beta = numerator_beta / denominator_beta
        
        return alpha, beta

    def calculate_probabilities(self, consensus_matrix, criteria_weights):
        """
        实现公式 (21): 计算每个方案的综合概率 Pr(A_l)。
        
        Input:
            consensus_matrix: (n, p) Numpy array. 最终群体共识矩阵 u_{c,j,l}^*。
            criteria_weights: (n,) Numpy array. 准则权重 v_j。
            
        Output:
            probabilities: (p,) Numpy array. 每个方案的得分/概率。
        """
        # Pr(A_l) = sum_j (v_j * u_{c,j,l})
        # 矩阵乘法: (n,) dot (n, p) -> (p,)
        probabilities = np.dot(criteria_weights, consensus_matrix)
        return probabilities

    def classify_alternatives(self, probabilities, alpha, beta):
        """
        执行三支决策分类 (Partitioning)。
        
        Input:
            probabilities: (p,) 方案概率。
            alpha, beta: 阈值。
            
        Output:
            classification: dict. 格式为 {方案索引: 'POS'/'BND'/'NEG'}
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
        实现公式 (22) 和 (23): TOPSIS 排序。
        
        Input:
            consensus_matrix: (n, p) Numpy array. u_{c,j,l}^*
            criteria_weights: (n,) Numpy array. v_j
            target_indices: list (optional). 仅对指定索引的方案进行排序 (例如仅 POS 域)。
                            如果为 None，则计算所有方案。
                            
        Output:
            closeness_coeffs: (p,) Numpy array. 贴近度系数 C_l^*。
                              不在 target_indices 中的方案值为 -1。
        """
        n, p = consensus_matrix.shape
        closeness_coeffs = -1 * np.ones(p) # 初始化为 -1
        
        if target_indices is None:
            target_indices = range(p)
            
        if len(target_indices) == 0:
            return closeness_coeffs

        # 1. 确定 PIS (Z+) 和 NIS (Z-)
        # z_j^+ = max_l {u_{c,j,l}}
        z_plus = np.max(consensus_matrix, axis=1) # (n,)
        # z_j^- = min_l {u_{c,j,l}}
        z_minus = np.min(consensus_matrix, axis=1) # (n,)
        
        # 2. 计算距离 D+ 和 D- (公式 22)
        # D_l = sqrt( sum_j v_j * (u - z)^2 )
        
        for l in target_indices:
            u_l = consensus_matrix[:, l] # (n,)
            
            # 距离计算 (注意公式中 v_j 是直接乘在平方项外的，不是加权欧氏距离的常规写法，需严格遵循公式)
            dist_sq_plus = np.sum(criteria_weights * ((u_l - z_plus) ** 2))
            d_plus = np.sqrt(dist_sq_plus)
            
            dist_sq_minus = np.sum(criteria_weights * ((u_l - z_minus) ** 2))
            d_minus = np.sqrt(dist_sq_minus)
            
            # 3. 计算相对贴近度 (公式 23)
            # C_l = D- / (D+ + D-)
            if d_plus + d_minus == 0:
                score = 0.0
            else:
                score = d_minus / (d_plus + d_minus)
            
            closeness_coeffs[l] = score
            
        return closeness_coeffs