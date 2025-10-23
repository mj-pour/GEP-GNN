# ================= GNNmodel.py =================
"""GNNmodel.py

Contains two model implementations that use edge attributes during message passing:

1. WeightedGCNModel: uses torch_geometric.nn.GCNConv which accepts `edge_weight`.
   We pass the normalized edge_attr (shape [E,1]) as `edge_weight` after flattening.

2. EdgeAttrGATModel: a custom attention-based MessagePassing layer that computes
   attention coefficients from node features (like GAT) and multiplies them by
   edge_attr to bias attention according to transition probabilities.

Utilities for training and evaluation that correctly move edge_attr to device
and keep batch handling are included.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax


class WeightedGCNModel(nn.Module):
    """GCN model that uses edge_attr as edge_weight for convolution.

    Expected Data fields:
      - x: LongTensor node feature indices (embedding happens inside)
      - edge_index: [2, E]
      - edge_attr: [E, 1] (normalized or raw counts) -> we use flattened as edge_weight
      - batch: [num_nodes] (when using DataLoader with batching)
    """
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList()
        in_dim = emb_dim
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.pool = global_mean_pool
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 2)
        )

    def forward(self, data):
        # x is indices -> embed
        x_idx = data.x.view(-1)
        x = self.embedding(x_idx)
        edge_index = data.edge_index
        # edge_attr -> edge_weight shape [E]
        edge_weight = None
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_weight = data.edge_attr.view(-1)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        g = self.pool(x, batch)
        out = self.mlp(g)
        return out


class EdgeAttrGATConv(MessagePassing):
    """Custom attention convolution that incorporates edge_attr multiplicatively
    into the attention coefficients.

    Implementation idea:
      - Linear transform of h: Wh
      - Compute attention e_ij = a^T [Wh_i || Wh_j]
      - Multiply e_ij by scalar from edge_attr (e.g., probability in [0,1])
      - Apply softmax over neighbors: alpha = softmax(e_ij)
      - Message = alpha * Wh_j
    """
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


class EdgeAttrGATModel(nn.Module):
    """Graph model using custom EdgeAttrGATConv layers.

    Expected Data fields same as WeightedGCNModel.
    """
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList()
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
        return out


# ---------- Training utilities (similar API as before) ----------

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        y = data.y.view(-1)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    ys = []
    preds = []
    probs = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            p = F.softmax(out, dim=1)[:,1].cpu().numpy()
            y = data.y.view(-1).cpu().numpy()
            pred = out.argmax(dim=1).cpu().numpy()
            ys.extend(y.tolist())
            preds.extend(pred.tolist())
            probs.extend(p.tolist())
    return np.array(ys), np.array(preds), np.array(probs)

# ================= SeqEncoder.py (reference) =================
# The SeqEncoder module (unchanged) should provide seq_to_graph that sets Data.edge_attr
# to normalized transition probabilities (shape [E,1]).

# ================= Example usage =================
# from GNNmodel import WeightedGCNModel, EdgeAttrGATModel, train_epoch, evaluate
# model = WeightedGCNModel(vocab_size=len(vocab))
# or
# model = EdgeAttrGATModel(vocab_size=len(vocab))
# Both accept Data objects where data.edge_attr contains edge weights.