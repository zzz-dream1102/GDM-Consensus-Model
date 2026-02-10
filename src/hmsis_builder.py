import numpy as np
import torch

class HMSISBuilder:
    def __init__(self, criteria_types, granular_size_G=3):
        """
        初始化 HMSIS 构建器。
        
        参数:
            criteria_types (list or np.array): 包含 'benefit' (效益型) 或 'cost' (成本型) 的列表，长度为 n (准则数)。
            granular_size_G (int): 第4尺度的粒度参数 G (默认 3)。
        """
        self.criteria_types = np.array(criteria_types)
        self.G = granular_size_G

    def calculate_scale1_utility(self, raw_data):
        """
        实现公式 (1): 多尺度信息的统一效用值转换 (Scale 1)。
        
        输入:
            raw_data: Numpy 数组，维度 (m, n, p)。代表 a_{i,j,l}。
                      m=专家数, n=准则数, p=方案数。
        输出:
            u_scale1: Numpy 数组，维度 (m, n, p)，值域 [0, 1]。
        """
        m, n, p = raw_data.shape
        u_scale1 = np.zeros_like(raw_data, dtype=np.float64)

        # 计算每个准则下的最大值和最小值 (跨所有专家和方案)
        # 保持维度 (1, n, 1) 以便后续广播计算
        min_vals = raw_data.min(axis=(0, 2), keepdims=True) 
        max_vals = raw_data.max(axis=(0, 2), keepdims=True) 
        range_vals = max_vals - min_vals
        
        # 防止除以零：如果极差为0，设为一个极小值
        range_vals[range_vals == 0] = 1e-9

        # 向量化实现公式 (1)
        for j in range(n):
            if self.criteria_types[j] == 'benefit':
                # 效益型: (值 - 最小值) / (最大值 - 最小值)
                u_scale1[:, j, :] = (raw_data[:, j, :] - min_vals[0, j, 0]) / range_vals[0, j, 0]
            elif self.criteria_types[j] == 'cost':
                # 成本型: (最大值 - 值) / (最大值 - 最小值)
                u_scale1[:, j, :] = (max_vals[0, j, 0] - raw_data[:, j, :]) / range_vals[0, j, 0]
            else:
                raise ValueError(f"未知的准则类型: {self.criteria_types[j]}")
                
        return u_scale1

    def calculate_scale3_ranking(self, u_scale1):
        """
        实现公式 (2) 和 (3): 生成模糊偏好关系并推导 Scale 3 (偏好排序)。
        
        输入:
            u_scale1: Numpy 数组，维度 (m, n, p)。
        输出:
            u_scale3: Numpy 数组，维度 (m, n, p)，包含排名 (1 到 p)。
        """
        m, n, p = u_scale1.shape
        
        # --- 公式 (2): Scale 2 模糊偏好关系 ---
        # 扩展维度以便广播计算: (m, n, p, 1) - (m, n, 1, p)
        # u_l 代表方案 l 的效用，形状 (m, n, p, 1)
        u_l = u_scale1[:, :, :, np.newaxis]
        # u_s 代表方案 s 的效用，形状 (m, n, 1, p)
        u_s = u_scale1[:, :, np.newaxis, :]
        
        # r 矩阵形状: (m, n, p, p)
        # r_{i,j,(l,s)} = 0.5 * (1 + u_l - u_s)
        r_scale2 = 0.5 * (1.0 + u_l - u_s)
        
        # --- 公式 (3): Scale 3 偏好排序 ---
        # Sco_{i,j,l} = Sum_{s!=l} r_{i,j,(l,s)}
        # 注意：对角线 s=l 时，r值为 0.5。求和后减去这个 0.5 即可，不用显式排除。
        scores = np.sum(r_scale2, axis=3) - 0.5 # 形状: (m, n, p)
        
        # 生成排名: 分数越高，排名越靠前 (Rank 1)
        # 使用 argsort 对分数的负值进行排序 (降序)
        temp_sort = np.argsort(-scores, axis=2)
        ranks = np.empty_like(temp_sort)
        
        # 使用 put_along_axis 将排名值填入对应的索引位置
        # 生成 1 到 p 的数组
        ranks_values = np.arange(1, p+1)[np.newaxis, np.newaxis, :]
        np.put_along_axis(ranks, temp_sort, ranks_values, axis=2)
        
        u_scale3 = ranks.astype(np.float64)
        return u_scale3

    def calculate_scale4_class(self, u_scale3):
        """
        实现公式 (4): 等价类划分 (Scale 4)。
        
        输入:
            u_scale3: Numpy 数组，维度 (m, n, p)，包含排名。
        输出:
            u_scale4: Numpy 数组，维度 (m, n, p)，包含类别索引。
        """
        # u_{i,j,l}^{(4,0)} = floor((u^{(3)} - 1) / G) + 1
        # 向下取整逻辑
        u_scale4 = np.floor((u_scale3 - 1) / self.G) + 1
        return u_scale4

    def construct_nodal_features(self, u_scale1, u_scale3, u_scale4):
        """
        实现公式 (5): 专家节点特征向量化。
        
        输入:
            u_scale1: (m, n, p)
            u_scale3: (m, n, p)
            u_scale4: (m, n, p)
        输出:
            h_0: PyTorch Tensor，维度 (m, 3 * n * p)。这是输入 GAT 的初始特征。
        """
        m, n, p = u_scale1.shape
        
        # 确保所有尺度维度一致
        assert u_scale1.shape == u_scale3.shape == u_scale4.shape
        
        # 根据公式 (5)，我们需要对每个准则 j 进行拼接：
        # 结构: 针对每个 j, 拼接 ([u1]_l || [u3]_l || [u4]_l)
        # 每个 j 的块大小应该是 3 * p
        
        # 步骤 1: 在新维度堆叠三个尺度
        # 堆叠后形状: (m, n, 3, p) -> (专家, 准则, 尺度类型, 方案)
        stacked = np.stack([u_scale1, u_scale3, u_scale4], axis=2)
        
        # 步骤 2: 展平最后两个维度 (尺度类型和方案)
        # 这里的 reshape 会按照内存顺序合并，即先排 scale1 的 p 个方案，再排 scale3...
        # 这符合公式中的 [u1] || [u3] || [u4]
        feature_blocks = stacked.reshape(m, n, 3 * p)
        
        # 步骤 3: 跨准则 j=1..n 进行拼接 (Concatenation)
        # 对应公式中的大符号 || (j=1 to n)
        # 最终形状 (m, n * 3 * p)
        h_0_numpy = feature_blocks.reshape(m, n * 3 * p)
        
        # 转换为 PyTorch Tensor 供后续 GAT 使用
        h_0 = torch.tensor(h_0_numpy, dtype=torch.float32)
        
        return h_0

    def process(self, raw_data):
        """
        执行 Stage 1 的完整流程。
        
        输入: 
            raw_data: (m, n, p) Numpy 数组
        输出: 
            h_0: (m, 3np) PyTorch Tensor
        """
        u1 = self.calculate_scale1_utility(raw_data)
        u3 = self.calculate_scale3_ranking(u1)
        u4 = self.calculate_scale4_class(u3)
        h_0 = self.construct_nodal_features(u1, u3, u4)
        return h_0