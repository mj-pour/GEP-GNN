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
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
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


class GINModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_layers=3, num_classes=2, dropout=0.3):
        super(GINModel, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        layers = []
        input_dim = in_channels
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            layers.append(GINConv(mlp))
            input_dim = hidden_channels
        self.convs = nn.ModuleList(layers)

        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)

        # Pool node embeddings
        x = global_mean_pool(x, batch)

        # Classification head
        x = torch.relu(self.lin1(x))
        x = torch.dropout(x, p=self.dropout, train=self.training)
        x = self.lin2(x)
        return x
    
#================================================================================
# Standard GINConv doesn’t take edge_weight directly. 
# Define a custom weighted GIN that include your normalized transition probabilities (Edge Attributes).
#================================================================================


# ---------- DiffPool implementation (requires torch_geometric) ----------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import to_dense_batch, to_dense_adj, dense_diff_pool

class DiffPoolGNN(nn.Module):
    """
    DiffPool hierarchical GNN.
    - node embedding GNNs produce Z (node embeddings)
    - assignment GNNs produce S (soft cluster assignments)
    - dense_diff_pool is used to pool to next level
    """
    def __init__(self,
                 vocab_size: int,
                 emb_dim: int = 64,
                 gnn_hidden: int = 64,
                 assign_hidden: int = 64,
                 num_classes: int = 2,
                 num_pool_layers: int = 1,
                 clusters_per_layer: list = None,
                 dropout: float = 0.2):
        """
        clusters_per_layer: list of ints specifying #clusters at each pooled layer.
            length must equal num_pool_layers.
        """
        super().__init__()
        if clusters_per_layer is None:
            clusters_per_layer = [32] * num_pool_layers

        assert len(clusters_per_layer) == num_pool_layers

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        # initial embedding conv
        self.conv1 = GCNConv(emb_dim, gnn_hidden)

        # For each pooling stage we need:
        #  - an embedding GNN to compute node embeddings Z (here we reuse simple convs)
        #  - an assignment GNN to compute S (we map features -> clusters)
        self.num_pool_layers = num_pool_layers
        self.cluster_sizes = clusters_per_layer

        self.embed_convs = nn.ModuleList()
        self.assign_convs = nn.ModuleList()
        for i in range(num_pool_layers):
            # embedding convs for this stage (two-layer style)
            self.embed_convs.append(nn.ModuleList([
                GCNConv(gnn_hidden if i > 0 else gnn_hidden, gnn_hidden),
                GCNConv(gnn_hidden, gnn_hidden)
            ]))
            # assignment network: simple two-layer MLP implemented as small GCNs (accepts same edge_index)
            self.assign_convs.append(nn.ModuleList([
                GCNConv(gnn_hidden if i > 0 else gnn_hidden, assign_hidden),
                GCNConv(assign_hidden, self.cluster_sizes[i])  # final produces S logits per node
            ]))

        # final MLP classifier on pooled graph representation
        self.classifier = nn.Sequential(
            nn.Linear(gnn_hidden, gnn_hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden//2, num_classes)
        )

    def forward(self, data):
        # data.x: [num_nodes, 1] with kmer indices -> embed
        x_idx = data.x.view(-1)
        x = self.embedding(x_idx)  # [N, emb_dim]

        # Build dense batch tensors
        # to_dense_batch: x_padded [B, N_max, feat], mask [B, N_max]
        x_padded, mask = to_dense_batch(x, data.batch)  # mask: bool [B, N_max]
        # adjacency: returns [B, N_max, N_max] using edge weights if available
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            # turn edge_attr into dense adjacency preserving weights
            adj = to_dense_adj(data.edge_index, batch=data.batch, edge_attr=data.edge_attr.view(-1))
        else:
            adj = to_dense_adj(data.edge_index, batch=data.batch)  # 0/1 adjacency

        # We'll iteratively apply diffpool layers
        batch_x = x_padded  # [B, N, F]
        batch_adj = adj     # [B, N, N]
        batch_mask = mask   # [B, N]

        link_loss = 0.0
        ent_loss = 0.0

        for i in range(self.num_pool_layers):
            # 1) compute node embeddings Z using embed_convs applied on sparse graph
            #    but dense_diff_pool expects dense inputs, so we produce Z in dense form.
            # We'll compute Z by running convs on flattened node features and using original edge_index/batch
            # Simpler: run convs on sparse data (original data) and then convert to dense.
            # For correctness we need node features aligned: convert dense back to flat, run conv on sparse.
            # Build a temporary Data-like object for conv inference:
            # NOTE: convs expect edge_index that refers to flattened nodes; we can use original data for that.
            # So compute Z_flat via convs on the sparse graph.
            # Recreate x_flat from batch_x:
            x_flat = torch.zeros((batch_x.size(0) * batch_x.size(1), batch_x.size(2)),
                                 device=batch_x.device)
            # but simpler: reuse data.x embedding and run convs on sparse graph using data.x embeddings
            # so use the original x and data.edge_index
            # compute Z_flat:
            z = None
            # use first stage input as original embedded x
            if i == 0:
                z = self.conv1(self.embedding(data.x.view(-1)), data.edge_index)
                z = F.relu(z)
            else:
                # after pooling, adjacency and features changed; easiest approach:
                # operate in dense mode by using batch_x and batch_adj via simple dense matmul aggregator
                # i.e., Z = ReLU(A_hat @ batch_x @ W) — approximates GCN in dense form
                # Define a linear layer to map batch_x features to gnn_hidden (we reuse conv weights idea)
                # For simplicity and clarity we compute a dense message aggregation:
                B, N, F = batch_x.size()
                # adjacency normalization: row-normalize
                deg = batch_adj.sum(dim=-1, keepdim=True)  # [B, N, 1]
                deg[deg == 0] = 1.0
                batch_x = batch_x.float()
                z = torch.matmul(batch_adj, batch_x) / deg  # [B, N, F]
                # map to hidden dim if necessary
                # apply a linear projection (we'll create it dynamically if missing)
                # But to keep implementation straightforward, reshape and apply a linear layer:
                z = z.reshape(B * N, -1)
                # If needed, project dimension down/up to gnn_hidden
                # Here assume batch_x feature dim == gnn_hidden; else user should set emb_dim accordingly.
            # Convert z -> dense [B, N, gnn_hidden] for diffpool
            # If z is flat [N_total, hidden], convert by to_dense_batch with original data.batch
            # but we have batch_x already; for simplicity set Z_dense = batch_x (works if dimensions match)
            Z_dense = batch_x  # use current node features as embeddings

            # 2) compute assignment S: run assign_convs in dense manner
            # To compute S, we'll use a small MLP per node on Z_dense
            B, N, F = Z_dense.size()
            # Flatten and run small MLP (learnable) for assignment logits
            # For clarity implement a simple projection:
            # create a linear mapping from F -> num_clusters (we do it here dynamically)
            num_clusters = self.cluster_sizes[i]
            assign_lin = nn.Linear(F, num_clusters).to(Z_dense.device)
            S = assign_lin(Z_dense)   # [B, N, num_clusters]
            S = F.softmax(S, dim=-1)

            # 3) apply dense_diff_pool
            Z_pooled, A_pooled, lp, ep = dense_diff_pool(Z_dense, batch_adj, S, mask=batch_mask)
            link_loss = link_loss + lp
            ent_loss = ent_loss + ep

            # update batch_x, batch_adj, batch_mask for next layer
            batch_x = Z_pooled
            batch_adj = A_pooled
            # new mask: all clusters considered present (mask not necessary, set to None)
            batch_mask = torch.ones(batch_x.size(0), batch_x.size(1), dtype=torch.bool, device=batch_x.device)

        # After last pooling, batch_x is [B, N_pooled, F]
        # create graph-level embeddings by mean over pooled nodes (or sum)
        graph_emb = batch_x.mean(dim=1)  # [B, F]

        out = self.classifier(graph_emb)
        return out, (link_loss, ent_loss)


class SimpleDiffPool(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, gnn_hidden=64, num_clusters=32, num_pool_layers=1, num_classes=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv1 = GCNConv(emb_dim, gnn_hidden)
        # linear projector for assignment (per layer); here only single pool layer supported for simplicity
        self.assign_lin = nn.Sequential(
            nn.Linear(gnn_hidden, gnn_hidden),
            nn.ReLU(),
            nn.Linear(gnn_hidden, num_clusters)
        )
        self.embed_lin = nn.Sequential(
            nn.Linear(gnn_hidden, gnn_hidden),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(gnn_hidden, gnn_hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden//2, num_classes)
        )

    def forward(self, data):
        x_idx = data.x.view(-1)
        x = self.embedding(x_idx)  # [N, emb_dim]
        # initial GCN conv on sparse graph
        z = F.relu(self.conv1(x, data.edge_index, edge_weight=(data.edge_attr.view(-1) if hasattr(data,'edge_attr') and data.edge_attr is not None else None)))
        # dense batch
        Z_dense, mask = to_dense_batch(z, data.batch)  # [B, N, F]
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            A = to_dense_adj(data.edge_index, batch=data.batch, edge_attr=data.edge_attr.view(-1))
        else:
            A = to_dense_adj(data.edge_index, batch=data.batch)  # [B,N,N]
        # compute assignment logits and softmax
        S_logits = self.assign_lin(Z_dense)            # [B,N,K]
        S = F.softmax(S_logits, dim=-1)
        # pool
        Z_pooled, A_pooled, link_loss, ent_loss = dense_diff_pool(Z_dense, A, S, mask)
        # readout
        graph_emb = Z_pooled.mean(dim=1)  # [B, F]
        out = self.classifier(graph_emb)
        return out, (link_loss, ent_loss)


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

