import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    实现公式 (6) 和 (7): 单层图注意力网络 (GAT)。
    """
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        """
        参数:
            in_features: 输入特征维度 (对应 h^{(q-1)} 的维度)
            out_features: 输出特征维度 (对应 h^{(q)} 的维度)
            dropout: Dropout 概率
            alpha: LeakyReLU 的负斜率参数
        """
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        # W^(q): 线性变换权重矩阵
        # 维度: (in_features, out_features)
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # a^(q): 注意力向量
        # 维度: (2 * out_features, 1) 因为输入是拼接的 [Wh_i || Wh_k]
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj_mask):
        """
        前向传播逻辑。
        
        输入:
            h: 输入特征矩阵 h^{(q-1)}
               维度: (m, in_features)
            adj_mask: 邻接矩阵掩码 M^{(t)}，用于定义邻域 N_i
               维度: (m, m)，如果 i, k 有连接则为 1 (或 True)，否则为 0
        
        输出:
            h_prime: 更新后的特征矩阵 h^{(q)}
               维度: (m, out_features)
        """
        m = h.size(0)

        # 1. 线性变换: Wh = h * W
        # 维度: (m, in_features) x (in_features, out_features) -> (m, out_features)
        Wh = torch.mm(h, self.W)

        # 2. 准备注意力机制的输入
        # 我们需要计算所有 (i, k) 对的 [Wh_i || Wh_k]
        # a_input 维度构建逻辑:
        # Wh_repeated_in_chunks: (m * m, out_features) -> 对应 Wh_i
        # Wh_repeated_alternating: (m * m, out_features) -> 对应 Wh_k
        Wh_repeat_i = Wh.repeat_interleave(m, dim=0)
        Wh_repeat_k = Wh.repeat(m, 1)
        
        # 拼接: [Wh_i || Wh_k]
        # 维度: (m * m, 2 * out_features)
        all_combinations = torch.cat([Wh_repeat_i, Wh_repeat_k], dim=1)

        # 3. 计算注意力系数 (公式 6 分子部分的核心)
        # e_ik = LeakyReLU(a^T * [Wh_i || Wh_k])
        # 维度: (m * m, 1)
        e = self.leakyrelu(torch.matmul(all_combinations, self.a))
        
        # 重塑为 (m, m) 矩阵形式
        e = e.view(m, m)

        # 4. Masked Attention (只关注邻域 N_i)
        # 对于没有连接的节点，将注意力分数设为负无穷，以便 Softmax 后为 0
        zero_vec = -9e15 * torch.ones_like(e)
        # 确保 adj_mask 是布尔或 0/1 (1表示存在边)
        attention = torch.where(adj_mask > 0, e, zero_vec)

        # 5. Softmax 归一化 (公式 6)
        # alpha_ik = exp(e_ik) / sum(exp(e_ir))
        # 维度: (m, m)
        attention = F.softmax(attention, dim=1)
        
        # 保存注意力系数供后续可能的分析
        self.attn_coefficients = attention

        # Dropout
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 6. 邻域聚合 (公式 7)
        # h_i^(q) = sigma( sum(alpha_ik * Wh_k) )
        # 矩阵乘法: (m, m) x (m, out_features) -> (m, out_features)
        h_prime = torch.matmul(attention, Wh)

        # 应用激活函数 sigma (通常为 ELU 或 ReLU，此处使用 ELU 以符合 GAT 惯例)
        return F.elu(h_prime)

class TrustCompletionModel(nn.Module):
    """
    对应 Phase 2: 完整的信任网络补全模型。
    包含多层 GAT 和最终的链接预测层。
    """
    def __init__(self, input_dim, hidden_dim, output_embed_dim, num_heads=1, lambda_reg=0.01):
        """
        参数:
            input_dim: 初始特征维度 (3 * n * p)
            hidden_dim: GAT 中间层维度
            output_embed_dim: 最终嵌入 h^{(L)} 的维度
            num_heads: 多头注意力的头数 (为简化代码，此处基础实现默认为 1，若需多头需调整 concat 逻辑)
            lambda_reg: L2 正则化系数
        """
        super(TrustCompletionModel, self).__init__()
        self.lambda_reg = lambda_reg

        # 定义 GAT 层 (假设 L=2，即两层 GAT)
        # 第一层: 输入 -> 隐藏
        self.gat1 = GraphAttentionLayer(input_dim, hidden_dim)
        # 第二层: 隐藏 -> 输出嵌入 (h^{(L)})
        self.gat2 = GraphAttentionLayer(hidden_dim, output_embed_dim)

        # 定义链接预测层的权重向量 omega (公式 8)
        # 输入是 [h_i || h_k]，所以维度是 2 * output_embed_dim
        # 输出是标量 (信任值)
        self.omega = nn.Parameter(torch.zeros(size=(2 * output_embed_dim, 1)))
        nn.init.xavier_uniform_(self.omega.data, gain=1.414)

    def forward(self, h_0, adj_mask):
        """
        前向传播: 生成嵌入并预测完整信任矩阵。
        
        输入:
            h_0: 初始特征矩阵 (m, input_dim)
            adj_mask: 初始信任掩码 (m, m)
        输出:
            predicted_T: 预测的完整信任矩阵 (m, m)
            h_L: 最终节点嵌入
        """
        # --- GAT 特征提取 ---
        # Layer 1
        h_1 = self.gat1(h_0, adj_mask)
        # Layer 2 (Final Layer L)
        h_L = self.gat2(h_1, adj_mask)

        # --- 链接预测 (公式 8) ---
        # hat_t_ik = Sigmoid(omega^T * [h_i^(L) || h_k^(L)])
        m = h_L.size(0)
        
        # 构造所有对应的拼接向量
        h_repeat_i = h_L.repeat_interleave(m, dim=0)
        h_repeat_k = h_L.repeat(m, 1)
        combined = torch.cat([h_repeat_i, h_repeat_k], dim=1) # (m*m, 2*out_dim)

        # 线性变换 + Sigmoid
        pred_scores = torch.matmul(combined, self.omega) # (m*m, 1)
        pred_matrix = torch.sigmoid(pred_scores).view(m, m)

        return pred_matrix, h_L

    def compute_loss(self, pred_matrix, true_matrix, obs_mask):
        """
        实现公式 (9): 损失函数。
        
        输入:
            pred_matrix: 预测的信任矩阵 hat_t (m, m)
            true_matrix: 真实的/初始的信任矩阵 t (m, m)
            obs_mask: 观测掩码 M (m, m)，已知信任的位置为 1
        输出:
            total_loss: 标量 Tensor
        """
        # 第一部分: Masked MSE
        # sum_{i,k} M_{ik} (hat_t - t)^2
        diff = pred_matrix - true_matrix
        squared_diff = diff ** 2
        mse_loss = torch.sum(obs_mask * squared_diff) / (torch.sum(obs_mask) + 1e-9) # 取平均更稳定

        # 第二部分: L2 正则化 (针对权重矩阵 W)
        # lambda * sum ||W||^2
        l2_reg = 0.0
        # 遍历所有 GAT 层收集 W 的 L2 范数
        l2_reg += torch.sum(self.gat1.W ** 2)
        l2_reg += torch.sum(self.gat2.W ** 2)
        
        total_loss = mse_loss + self.lambda_reg * l2_reg
        
        return total_loss