import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import dense_diff_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj

class WeightedGINDense(nn.Module):
    def __init__(self, mlp, eps=0.0, train_eps=True):
        super().__init__()
        self.mlp = mlp
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

    def forward(self, x, adj):
        # adj: [B, N, N], x: [B, N, F]
        I = torch.eye(adj.size(-1), device=adj.device).unsqueeze(0)
        adj_with_self = adj + I
        deg = adj_with_self.sum(-1, keepdim=True)
        out = torch.bmm(adj_with_self, x) / (deg + 1e-6)
        out = (1 + self.eps) * x + out
        return self.mlp(out)


class model(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128,
                 cluster_ratio1=0.25, cluster_ratio2=0.10,
                 num_classes=2, dropout=0.3):
        super(model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.cluster_ratio1 = cluster_ratio1
        self.cluster_ratio2 = cluster_ratio2

        # -------- Shared MLP Builder --------
        def make_mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )

        # -------- Block 1 (Before 1st DiffPool) --------
        self.gnn_embed1 = WeightedGINDense(make_mlp(emb_dim, hidden_dim))
        self.gnn_embed2 = WeightedGINDense(make_mlp(hidden_dim, hidden_dim))
        self.gnn_assign1 = WeightedGINDense(make_mlp(emb_dim, hidden_dim))

        self.norm1 = nn.LayerNorm(hidden_dim)

        # -------- Block 2 (After 1st DiffPool) --------
        self.gnn_embed3 = WeightedGINDense(make_mlp(hidden_dim, hidden_dim))
        self.gnn_embed4 = WeightedGINDense(make_mlp(hidden_dim, hidden_dim))
        self.gnn_embed5 = WeightedGINDense(make_mlp(hidden_dim, hidden_dim))
        self.gnn_assign2 = WeightedGINDense(make_mlp(hidden_dim, hidden_dim))

        self.norm2 = nn.LayerNorm(hidden_dim)

        # -------- Block 3 (After 2nd DiffPool) --------
        self.gnn_embed6 = WeightedGINDense(make_mlp(hidden_dim, hidden_dim))
        self.gnn_embed7 = WeightedGINDense(make_mlp(hidden_dim, hidden_dim))
        self.gnn_embed8 = WeightedGINDense(make_mlp(hidden_dim, hidden_dim))

        self.norm3 = nn.LayerNorm(hidden_dim)

        # -------- Readout --------
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, data):
        x_idx = data.x.view(-1)
        x = self.embedding(x_idx)
        edge_index, edge_attr = data.edge_index, data.edge_attr
        if edge_attr is not None:
            edge_attr = edge_attr.view(-1)

        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        x_dense, mask = to_dense_batch(x, batch)
        adj_dense = to_dense_adj(edge_index, batch, edge_attr)

        # Dynamically determine cluster sizes
        N_max = x_dense.size(1)
        assign_dim1 = max(2, int(self.cluster_ratio1 * N_max))
        assign_dim2 = max(2, int(self.cluster_ratio2 * assign_dim1))

        # ========== Block 1 ==========
        z = F.relu(self.gnn_embed1(x_dense, adj_dense))
        z = F.relu(self.gnn_embed2(z, adj_dense))
        s = F.softmax(self.gnn_assign1(x_dense, adj_dense)[..., :assign_dim1], dim=-1)
        z, adj, l1, e1 = dense_diff_pool(z, adj_dense, s, mask)

        z = self.norm1(z)
        z = F.dropout(z, p=self.dropout, training=self.training)

        # ========== Block 2 ==========
        z = F.relu(self.gnn_embed3(z, adj))
        z = F.relu(self.gnn_embed4(z, adj))
        z = F.relu(self.gnn_embed5(z, adj))
        s = F.softmax(self.gnn_assign2(z, adj)[..., :assign_dim2], dim=-1)
        z, adj, l2, e2 = dense_diff_pool(z, adj, s)

        z = self.norm2(z)
        z = F.dropout(z, p=self.dropout, training=self.training)

        # ========== Block 3 ==========
        z = F.relu(self.gnn_embed6(z, adj))
        z = F.relu(self.gnn_embed7(z, adj))
        z = F.relu(self.gnn_embed8(z, adj))
        z = self.norm3(z)

        # Graph-level readout (mean over remaining nodes)
        g = z.mean(dim=1)
        out = self.mlp_out(g)
        return F.softmax(out, dim=1), (l1 + l2 + e1 + e2)
