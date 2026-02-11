import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    实现公式 (6) 和 (7): 单层图注意力网络 (GAT)。
    """
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        # W^(q): 线性变换权重矩阵
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # a^(q): 注意力向量
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj_mask):
        m = h.size(0)
        Wh = torch.mm(h, self.W)

        Wh_repeat_i = Wh.repeat_interleave(m, dim=0)
        Wh_repeat_k = Wh.repeat(m, 1)
        all_combinations = torch.cat([Wh_repeat_i, Wh_repeat_k], dim=1)

        e = self.leakyrelu(torch.matmul(all_combinations, self.a))
        e = e.view(m, m)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_mask > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        
        self.attn_coefficients = attention
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)

class TrustCompletionModel(nn.Module):
    """
    对应 Phase 2: 完整的信任网络补全模型。
    """
    def __init__(self, input_dim, hidden_dim, output_embed_dim, num_heads=1, lambda_reg=0.01):
        super(TrustCompletionModel, self).__init__()
        self.lambda_reg = lambda_reg

        self.gat1 = GraphAttentionLayer(input_dim, hidden_dim)
        self.gat2 = GraphAttentionLayer(hidden_dim, output_embed_dim)

        self.omega = nn.Parameter(torch.zeros(size=(2 * output_embed_dim, 1)))
        nn.init.xavier_uniform_(self.omega.data, gain=1.414)

    def forward(self, h_0, adj_mask):
        h_1 = self.gat1(h_0, adj_mask)
        h_L = self.gat2(h_1, adj_mask)

        m = h_L.size(0)
        h_repeat_i = h_L.repeat_interleave(m, dim=0)
        h_repeat_k = h_L.repeat(m, 1)
        combined = torch.cat([h_repeat_i, h_repeat_k], dim=1)

        pred_scores = torch.matmul(combined, self.omega)
        pred_matrix = torch.sigmoid(pred_scores).view(m, m)

        return pred_matrix, h_L

    def compute_loss(self, pred_matrix, true_matrix, obs_mask):
        """
        [AUDIT FIX] Strictly implements Eq. 9 (Sum of Squared Errors).
        Removed the division by count (MSE -> SSE).
        """
        # 第一部分: Squared Error Sum (SSE)
        # sum_{i,k} M_{ik} (hat_t - t)^2
        diff = pred_matrix - true_matrix
        squared_diff = diff ** 2
        # FIX: Removed denominator to match formula strictly
        sse_loss = torch.sum(obs_mask * squared_diff) 

        # 第二部分: L2 正则化
        l2_reg = 0.0
        l2_reg += torch.sum(self.gat1.W ** 2)
        l2_reg += torch.sum(self.gat2.W ** 2)
        
        total_loss = sse_loss + self.lambda_reg * l2_reg
        
        return total_loss