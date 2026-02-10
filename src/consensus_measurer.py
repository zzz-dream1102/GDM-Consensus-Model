import numpy as np

class ConsensusMeter:
    """
    实现 Phase 4: 共识测量与冲突检测 (Consensus Measurement and Conflict Detection)。
    """
    
    def __init__(self):
        pass

    def aggregate_opinions(self, opinions, weights):
        """
        实现公式 (12): 群体意见聚合 (Group Opinion Aggregation)。
        
        输入:
            opinions: Numpy 数组，维度 (m, n, p)。代表个体评价 u_{i,j,l}^{(1,t)}。
                      m=专家数, n=准则数, p=方案数。
            weights: Numpy 数组，维度 (m,)。代表专家权重 W_i^{(t)} (需满足和为1)。
            
        输出:
            group_opinion: Numpy 数组，维度 (n, p)。代表群体共识评价 u_{c,j,l}^{(t)}。
        """
        # 确保 weights 形状正确以便广播: (m, 1, 1)
        # u_{c,j,l} = sum_i (W_i * u_{i,j,l})
        # tensordot 或者简单的乘法求和都可以
        
        w_reshaped = weights[:, np.newaxis, np.newaxis] # (m, 1, 1)
        
        # 加权求和，沿着专家维度 (axis=0)
        group_opinion = np.sum(opinions * w_reshaped, axis=0)
        
        return group_opinion

    def calculate_consensus_levels(self, individual_opinions, group_opinion):
        """
        实现公式 (13) 和 (14): 共识水平评估 (Consensus Level Assessment)。
        
        输入:
            individual_opinions: Numpy 数组，维度 (m, n, p)。
            group_opinion: Numpy 数组，维度 (n, p)。
            
        输出:
            cl_values: Numpy 数组，维度 (m,)。代表每个专家的个体共识度 CL_i^{(t)}。
            gcl_value: 标量 float。代表群体共识度 GCL^{(t)}。
        """
        m, n, p = individual_opinions.shape
        
        # --- 公式 (13): 个体共识度 CL_i ---
        # CL_i = 1 - (1/np) * sum_j sum_l |u_{i,j,l} - u_{c,j,l}|
        
        # 扩展 group_opinion 以匹配 m 维度: (1, n, p) -> (m, n, p) (广播自动处理)
        diff = np.abs(individual_opinions - group_opinion)
        
        # 对每个专家 i，求所有 j, l 的和
        sum_diff = np.sum(diff, axis=(1, 2)) # 维度: (m,)
        
        # 计算 CL_i
        cl_values = 1.0 - (sum_diff / (n * p))
        
        # --- 公式 (14): 群体共识度 GCL ---
        # GCL = (1/m) * sum_i CL_i
        gcl_value = np.mean(cl_values)
        
        return cl_values, gcl_value

    def detect_conflicts(self, gcl_value, threshold=0.85):
        """
        实现公式 (15) 后的逻辑: 冲突识别与状态初始化。
        判断是否需要进入下一阶段 (Consensus Reaching Process)。
        
        输入:
            gcl_value: 当前的群体共识度 GCL^{(t)}。
            threshold: 共识阈值 \bar{\theta} (默认为 0.85，可调整)。
            
        输出:
            needs_adjustment: 布尔值。如果 GCL <= threshold，则返回 True (需要调整)。
        """
        # 如果 GCL <= theta, 进入共识达成阶段 (Stage 5)
        return gcl_value <= threshold

    def get_state(self, cl_values, gcl_value):
        """
        辅助函数: 获取强化学习的状态 S_t = [GCL, CL_min]。
        
        输入:
            cl_values: (m,) 数组
            gcl_value: float
        输出:
            state: Numpy 数组 [GCL, CL_min]
        """
        cl_min = np.min(cl_values)
        return np.array([gcl_value, cl_min])