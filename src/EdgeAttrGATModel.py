import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax


class EdgeAttrGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, concat=True, negative_slope=0.2, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.concat = concat
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        # attention vector a for source and target
        self.att_src = nn.Parameter(torch.Tensor(out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(out_channels))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.att_dst.unsqueeze(0))

    def forward(self, x, edge_index, edge_attr=None):
        # x: [N, in_channels]
        H = self.lin(x)  # [N, out_channels]
        # add self loops to edge_index if not present because attention often uses them
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # If edge_attr provided, we must also add self-loop attrs (set to 1)
        if edge_attr is None:
            edge_attr_proc = None
        else:
            # original edge_attr shape [E, 1] -> flatten
            ea = edge_attr.view(-1)
            # For added self-loops, append ones
            ea = torch.cat([ea, torch.ones(x.size(0), device=ea.device)], dim=0)
            edge_attr_proc = ea
        return self.propagate(edge_index, x=H, edge_attr=edge_attr_proc)

    def message(self, x_i, x_j, edge_index_i, edge_attr):
        # x_i: target node features [E, out_channels]
        # x_j: source node features [E, out_channels]
        # compute attention score
        alpha = (x_i * self.att_dst).sum(dim=-1) + (x_j * self.att_src).sum(dim=-1)  # [E]
        alpha = self.leaky_relu(alpha)
        # incorporate edge_attr multiplicatively if available
        if edge_attr is not None:
            alpha = alpha * edge_attr.view(-1)
        # normalize attention coefficients for each target node
        alpha = softmax(alpha, edge_index_i)
        # scale messages
        return x_j * alpha.view(-1, 1)


class model(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList()
        self.dropout = dropout
        in_dim = emb_dim
        for i in range(num_layers):
            out_dim = hidden_dim
            self.convs.append(EdgeAttrGATConv(in_dim, out_dim))
            in_dim = out_dim
        self.pool = global_mean_pool
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 2)
        )

    def forward(self, data):
        x_idx = data.x.view(-1)
        x = self.embedding(x_idx)
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        # flatten edge_attr if present
        if edge_attr is not None:
            edge_attr = edge_attr.view(-1)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        g = self.pool(x, batch)
        out = self.mlp(g)
        return F.softmax(out, dim=1)