import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # Eq. 6: W^(q) learnable weight matrix
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Eq. 6: v^(q) attention mechanism weight vector (Modified from self.a)
        self.v_att = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.v_att.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj_mask):
        m = h.size(0)
        Wh = torch.mm(h, self.W)
        Wh_repeat_i = Wh.repeat_interleave(m, dim=0)
        Wh_repeat_k = Wh.repeat(m, 1)
        all_combinations = torch.cat([Wh_repeat_i, Wh_repeat_k], dim=1)
        
        # Applying attention vector v^(q)
        e = self.leakyrelu(torch.matmul(all_combinations, self.v_att))
        e = e.view(m, m)
        
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_mask > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        
        self.attn_coefficients = attention
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)

class TrustCompletionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_embed_dim, num_heads=1, lambda_reg=0.01):
        super(TrustCompletionModel, self).__init__()
        self.lambda_reg = lambda_reg
        self.gat1 = GraphAttentionLayer(input_dim, hidden_dim)
        self.gat2 = GraphAttentionLayer(hidden_dim, output_embed_dim)
        
        # Eq. 8: w_pred weight vector for link prediction (Modified from self.omega)
        self.w_pred = nn.Parameter(torch.zeros(size=(2 * output_embed_dim, 1)))
        nn.init.xavier_uniform_(self.w_pred.data, gain=1.414)

    def forward(self, h_0, adj_mask):
        h_1 = self.gat1(h_0, adj_mask)
        h_L = self.gat2(h_1, adj_mask)
        m = h_L.size(0)
        h_repeat_i = h_L.repeat_interleave(m, dim=0)
        h_repeat_k = h_L.repeat(m, 1)
        combined = torch.cat([h_repeat_i, h_repeat_k], dim=1)
        
        # Prediction using w_pred
        pred_scores = torch.matmul(combined, self.w_pred)
        pred_matrix = torch.sigmoid(pred_scores).view(m, m)
        return pred_matrix, h_L

    def compute_loss(self, pred_matrix, true_matrix, obs_mask):
        # Eq. 9: Masked Sum of Squared Errors (SSE)
        diff = pred_matrix - true_matrix
        sse_loss = torch.sum(obs_mask * (diff ** 2)) 

        # L2 Regularization
        l2_reg = torch.sum(self.gat1.W ** 2) + torch.sum(self.gat2.W ** 2)
        return sse_loss + self.lambda_reg * l2_reg