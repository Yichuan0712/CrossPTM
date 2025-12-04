import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class CoattentiontraceptionBlock(nn.Module):
    def __init__(self, args):
        super(CoattentiontraceptionBlock, self).__init__()

        self.bcn = CoAttentionLayer(
            v_dim=1280,
            pLMs_q_dim=1280,
            h_dim=1280,
            layer=1
        )

        # MLP Classifier
        self.mlp_classifier = MLPDecoder(
            in_dim=1280,
            hidden_dim=640,
            out_dim=320,
            binary=2
        )

        self.alpha = nn.Parameter(torch.rand(1))

    def attention_pooling(self, v, q, att_map):
        att_map = att_map.squeeze(-1)
        fusion_logits = torch.einsum('bvk,bvq,bqk->bvk', (v, att_map, q))
        return fusion_logits

    def forward(self, protein_embedding, task_embedding):
        v2, q2, att2 = self.bcn(protein_embedding, task_embedding)
        weighted_att_maps2 = att2 * (1 - self.alpha)
        fusion_logits2 = self.attention_pooling(v2, q2, weighted_att_maps2)

        f = fusion_logits2

        # Final score calculation using MLP
        score = self.mlp_classifier(f)

        return score


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)
        self.fc3 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.ln1(F.relu(self.fc1(x)))
        x = self.ln2(F.relu(self.fc2(x)))
        x = self.fc3(x[:, 1:-1, :])
        return x


class FullyConnectedNetwork(nn.Module):
    """A class for a fully connected network with optional activation and dropout."""

    def __init__(self, dims, activation='ReLU', dropout=0.0):
        super(FullyConnectedNetwork, self).__init__()

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if activation:
                    layers.append(getattr(nn, activation)(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SpatialDepthWiseConvolution(nn.Module):
    def __init__(self, head_dim: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels=head_dim, out_channels=head_dim, kernel_size=(kernel_size,),
                              padding=(kernel_size - 1,), groups=head_dim)

    def forward(self, x: torch.Tensor):
        batch_size, heads, seq_len, head_dim = x.shape
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size * heads, head_dim, seq_len)
        x = self.conv(x)
        if self.kernel_size > 1:
            x = x[:, :, :-(self.kernel_size - 1)]
        x = x.view(batch_size, heads, head_dim, seq_len)
        x = x.permute(0, 1, 3, 2)
        return x


class MHAtt(nn.Module):
    def __init__(self, hidden_size, dropout, multi_head, use_conv_for_query=False, use_conv_for_key_value=False):
        super(MHAtt, self).__init__()
        hidden_size_head = int(hidden_size / multi_head)

        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_merge = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

        self.multi_head = multi_head
        self.hidden_size_head = hidden_size_head
        self.hidden_size = hidden_size

        self.use_conv_for_query = use_conv_for_query
        self.use_conv_for_key_value = use_conv_for_key_value
        self.num_heads = 8
        self.embed_dim = 1280
        self.head_dim = self.embed_dim // self.num_heads

        assert self.num_heads % 4 == 0, "Invalid number of heads. Tranception requires the number of heads to be a multiple of 4."
        self.num_heads_per_kernel_size = self.num_heads // 4

        # 根据配置创建Query卷积
        if self.use_conv_for_query:
            self.query_depthwiseconv = nn.ModuleDict()
            for kernel_idx, kernel in enumerate([3, 5, 7]):
                self.query_depthwiseconv[str(kernel_idx)] = SpatialDepthWiseConvolution(
                    self.head_dim, kernel
                )
        # 根据配置创建Key,Value卷积
        elif self.use_conv_for_key_value:
            self.key_depthwiseconv = nn.ModuleDict()
            self.value_depthwiseconv = nn.ModuleDict()
            for kernel_idx, kernel in enumerate([3, 5, 7]):
                self.key_depthwiseconv[str(kernel_idx)] = SpatialDepthWiseConvolution(
                    self.head_dim, kernel
                )
                self.value_depthwiseconv[str(kernel_idx)] = SpatialDepthWiseConvolution(
                    self.head_dim, kernel
                )

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.multi_head,
            self.hidden_size_head
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.multi_head,
            self.hidden_size_head
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.multi_head,
            self.hidden_size_head
        ).transpose(1, 2)

        # Query卷积
        if self.use_conv_for_query:
            query_list = [q[:, :self.num_heads_per_kernel_size, :, :]]
            for kernel_idx in range(3):
                start_idx = (kernel_idx + 1) * self.num_heads_per_kernel_size
                end_idx = (kernel_idx + 2) * self.num_heads_per_kernel_size
                query_list.append(
                    self.query_depthwiseconv[str(kernel_idx)](
                        q[:, start_idx:end_idx, :, :]
                    )
                )
            q = torch.cat(query_list, dim=1)

        # Key和Value卷积
        if self.use_conv_for_key_value:
            key_list = [k[:, :self.num_heads_per_kernel_size, :, :]]
            value_list = [v[:, :self.num_heads_per_kernel_size, :, :]]

            for kernel_idx in range(3):
                start_idx = (kernel_idx + 1) * self.num_heads_per_kernel_size
                end_idx = (kernel_idx + 2) * self.num_heads_per_kernel_size

                key_list.append(
                    self.key_depthwiseconv[str(kernel_idx)](
                        k[:, start_idx:end_idx, :, :]
                    )
                )
                value_list.append(
                    self.value_depthwiseconv[str(kernel_idx)](
                        v[:, start_idx:end_idx, :, :]
                    )
                )

            k = torch.cat(key_list, dim=1)
            v = torch.cat(value_list, dim=1)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hidden_size
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e4)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


class FFN(nn.Module):
    def __init__(self, hidden_size, ff_size, dropout):
        super(FFN, self).__init__()
        self.network = FullyConnectedNetwork(
            dims=[hidden_size, ff_size, hidden_size],
            activation='ReLU',
            dropout=dropout
        )

    def forward(self, x):
        return self.network(x)


class SGA(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(SGA, self).__init__()

        # self.mhatt_self_x = MHAtt(hidden_size, dropout, multi_head=8)
        # self.mhatt_self_y = MHAtt(hidden_size, dropout, multi_head=8)

        self.mhatt_cross_xy = MHAtt(hidden_size, dropout, multi_head=8, use_conv_for_query=True,
                                    use_conv_for_key_value=False)
        self.mhatt_cross_yx = MHAtt(hidden_size, dropout, multi_head=8, use_conv_for_query=False,
                                    use_conv_for_key_value=True)

        self.ffn_x = FFN(hidden_size, ff_size=hidden_size, dropout=dropout)
        self.ffn_y = FFN(hidden_size, ff_size=hidden_size, dropout=dropout)

        self.dropout1_x = nn.Dropout(dropout)
        self.norm1_x = LayerNorm(hidden_size)
        self.dropout2_x = nn.Dropout(dropout)
        self.norm2_x = LayerNorm(hidden_size)
        self.dropout3_x = nn.Dropout(dropout)
        self.norm3_x = LayerNorm(hidden_size)

        self.dropout1_y = nn.Dropout(dropout)
        self.norm1_y = LayerNorm(hidden_size)
        self.dropout2_y = nn.Dropout(dropout)
        self.norm2_y = LayerNorm(hidden_size)
        self.dropout3_y = nn.Dropout(dropout)
        self.norm3_y = LayerNorm(hidden_size)

    def forward(self, x, y, x_mask, y_mask):
        # x = self.norm1_x(x + self.dropout1_x(
        #     self.mhatt_self_x(x, x, x, x_mask)
        # ))
        #
        # y = self.norm1_y(y + self.dropout1_y(
        #     self.mhatt_self_y(y, y, y, y_mask)
        # ))

        x = self.norm2_x(x + self.dropout2_x(
            self.mhatt_cross_xy(y, y, x, y_mask)
        ))

        y = self.norm2_y(y + self.dropout2_y(
            self.mhatt_cross_yx(x, x, y, x_mask)
        ))

        x = self.norm3_x(x + self.dropout3_x(
            self.ffn_x(x)
        ))

        y = self.norm3_y(y + self.dropout3_y(
            self.ffn_y(y)
        ))

        return x, y


class MCA_ED(nn.Module):
    def __init__(self, layer, hidden_size, dropout):
        super(MCA_ED, self).__init__()
        self.layer_stack = nn.ModuleList([SGA(hidden_size, dropout) for _ in range(layer)])

    def forward(self, x, y, x_mask, y_mask):
        for layer_module in self.layer_stack:
            x, y = layer_module(x, y, x_mask, y_mask)
        return x, y


class CoAttentionLayer(nn.Module):
    def __init__(self, v_dim, pLMs_q_dim, h_dim, layer, activation='ReLU', dropout=0.2, K=1):
        super(CoAttentionLayer, self).__init__()
        #
        # self.v_net = FullyConnectedNetwork([v_dim, h_dim], activation=activation, dropout=dropout)
        # self.q_net = FullyConnectedNetwork([pLMs_q_dim, h_dim], activation=activation, dropout=dropout)

        self.backbone = MCA_ED(layer, h_dim, dropout)

        self.att_net = nn.Linear(h_dim, K)
        self.proj_norm = LayerNorm(h_dim)

    def attention_pooling(self, v, q, att_map):
        att_map = att_map.squeeze(-1)
        fusion_logits = torch.einsum('bvk,bvq,bqk->bvk', (v, att_map, q))
        return fusion_logits

    def make_mask(self, feature):
        return (torch.sum(torch.abs(feature), dim=-1) == 0).unsqueeze(1).unsqueeze(2)

    def forward(self, v, q):
        v_mask = self.make_mask(v)
        q_mask = self.make_mask(q)

        # v = self.v_net(v)
        # q = self.q_net(q)

        v, q = self.backbone(v, q, v_mask, q_mask)

        att_scores = self.att_net(v.unsqueeze(2) * q.unsqueeze(1)).squeeze(-1)

        att_scores = att_scores.masked_fill(v_mask.squeeze(1).squeeze(1).unsqueeze(2).expand(-1, -1, 100), -1e4)

        att_scores = att_scores.masked_fill(q_mask.squeeze(1).squeeze(1).unsqueeze(1).expand(-1, 1026, -1), -1e4)

        att_maps = torch.softmax(att_scores, dim=-1).unsqueeze(-1)

        return v, q, att_maps
