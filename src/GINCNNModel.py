import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, to_dense_batch
    

class WeightedGINConv(MessagePassing):
    """
    A custom GIN layer that incorporates edge weights (edge_attr) 
    into the neighborhood aggregation step.
    """
    def __init__(self, mlp, eps=0.0, train_eps=True):
        super(WeightedGINConv, self).__init__(aggr='add')
        self.mlp = mlp
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

    def forward(self, x, edge_index, edge_attr=None):
        # Add self-loops to preserve node identity
        # Self-loops are given a default weight of 1.0
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr=edge_attr, fill_value=1.0, num_nodes=x.size(0)
        )

        # Message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Update with epsilon (self-weight)
        out = (1 + self.eps) * x + out
        return self.mlp(out)

    def message(self, x_j, edge_attr):
        # Each message is scaled by the edge weight
        if edge_attr is None:
            return x_j
        else:
            w = edge_attr.view(-1, 1)
            return w * x_j

class model(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, num_layers=3,
                  dropout=0.3, cnn_channels=[64, 128, 256]):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = dropout
        self.num_layers = num_layers

        # ---- GIN Layers ----
        layers = []
        input_dim = emb_dim
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            layers.append(WeightedGINConv(mlp))
            input_dim = hidden_dim
        self.convs = nn.ModuleList(layers)

        # ---- 1D CNN Layers ----
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=cnn_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_channels[0], out_channels=cnn_channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_channels[1], out_channels=cnn_channels[2], kernel_size=3, padding=1),
            nn.ReLU()
        )

        # ---- Fully Connected Layers ----
        self.fc = nn.Sequential(
            nn.Linear(cnn_channels[-1], cnn_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cnn_channels[-1] // 2, 2)
        )

    def forward(self, data):
        x_idx = data.x.view(-1)
        x = self.embedding(x_idx)
        edge_index = data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)

        # ---- GIN Feature Extraction ----
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            if i < len(self.convs) - 1 and self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # --- Group nodes by graph ---
        num_graphs = int(batch.max().item()) + 1
        max_nodes = torch.bincount(batch).max().item()
        hidden_dim = x.size(-1)

        x_padded = torch.zeros((num_graphs, max_nodes, hidden_dim), device=x.device)
        for i in range(num_graphs):
            nodes = x[batch == i]
            x_padded[i, :nodes.size(0)] = nodes

        
        # --- CNN expects [B, C_in, L] ---
        x_cnn = x_padded.permute(0, 2, 1)  # [batch, hidden_dim, num_nodes]
        x_feat = self.cnn(x_cnn)            # [batch, C_out, L_reduced]
        
        # ---- Global pooling over sequence length ----
        x_feat = F.adaptive_avg_pool1d(x_feat, 1).squeeze(-1)  # [batch, C_out]

        # ---- Fully connected classification ----
        out = self.fc(x_feat)
        return F.softmax(out, dim=1)