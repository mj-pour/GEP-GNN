import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class GINModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, num_layers=3, dropout=0.3):
        super(GINModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = dropout
        self.num_layers = num_layers

        layers = []
        input_dim = emb_dim
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            layers.append(GINConv(mlp))
            input_dim = hidden_dim
        self.convs = nn.ModuleList(layers)

        self.pool = global_mean_pool
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, data):
        x_idx = data.x.view(-1)
        x = self.embedding(x_idx)
        edge_index = data.edge_index

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            if i < len(self.convs) - 1 and self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        g = self.pool(x, batch)
        out = self.mlp(g)

        return F.softmax(out, dim=1)
    

class WeightedGINConv(MessagePassing):
    """
    A custom GIN layer that incorporates edge weights (edge_attr) 
    into the neighborhood aggregation step.
    """
    def __init__(self, mlp, eps=0.0, train_eps=True):
        super(WeightedGINConv, self).__init__(aggr='add')
        self.mlp = mlp

        # whether the eps parameter is learnable during training or treated as a fixed constant
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

    def forward(self, x, edge_index, edge_attr=None):
        # Add self-loops to preserve node identity
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr=edge_attr, fill_value=1.0, num_nodes=x.size(0)
        )
        # Calling Message function (core message passing engine from PyTorch Geometric)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        """
        When you call propagate(), It automatically handles:
            1. Message Generation
                Calls your message() function for each edge
                Passes relevant data: x_j (neighbor features), edge_attr (edge weights)
                Your message() function defines how to combine them

            2. Aggregation
                Uses the aggregation method you specified (aggr='add' in __init__)
                Sums all incoming messages for each node
                Could also use mean, max, etc.
        """
        # Update with epsilon (self-weight)
        out = (1 + self.eps) * x + out

        return self.mlp(out) # new_features = MLP((1 + ε) × self_features + Σ(edge_weight × neighbor_features))

    def message(self, x_j, edge_attr):
        # Each message is scaled by the edge weight
        if edge_attr is None:
            return x_j
        else:
            w = edge_attr.view(-1, 1)
            return w * x_j


class model_1(nn.Module):
    """Weighted GIN model that uses edge_attr as transition probabilities."""
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, num_layers=3, dropout=0.3):
        super(model_1, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = dropout
        self.num_layers = num_layers

        layers = []
        input_dim = emb_dim
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            layers.append(WeightedGINConv(mlp))
            input_dim = hidden_dim
        self.convs = nn.ModuleList(layers)

        self.lns = nn.ModuleList() 
        for l in range(self.num_layers-1): 
            self.lns.append(nn.LayerNorm(self.hidden_dim))  
        
        self.pool = global_mean_pool
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, data):
        # Node embedding
        x_idx = data.x.view(-1)
        x = self.embedding(x_idx)

        edge_index = data.edge_index
        edge_attr = data.edge_attr
        if edge_attr is not None:
            edge_attr = edge_attr.view(-1)

        # Convolutional layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Pooling and classification
        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        g = self.pool(x, batch)
        out = self.mlp(g)
        return F.softmax(out, dim=1)


class model_2(nn.Module):
    """Weighted GIN model with Layer Normalization between GNN layers and
        without dropout for MLP."""
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, num_layers=3, dropout=0.3):
        super(model_2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = dropout
        self.num_layers = num_layers

        layers = []
        norms = []
        input_dim = emb_dim
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            layers.append(WeightedGINConv(mlp))
            
            # Add LayerNorm for all layers EXCEPT the last one
            if i < num_layers - 1:
                norms.append(nn.LayerNorm(hidden_dim))
            else:
                norms.append(nn.Identity())  # No normalization for last layer
            
            input_dim = hidden_dim
        
        self.convs = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)

        self.pool = global_mean_pool
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, data):
        x_idx = data.x.view(-1)
        x = self.embedding(x_idx)

        edge_index = data.edge_index
        edge_attr = data.edge_attr
        if edge_attr is not None:
            edge_attr = edge_attr.view(-1)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        g = self.pool(x, batch)
        out = self.mlp(g)
        return F.softmax(out, dim=1)

class model_3(nn.Module):
    """Weighted GIN model with Pre-Normalization and
        without ReLU in forward pass of GIN and
        without dropout for MLP."""
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, num_layers=3, dropout=0.3):
        super(model_3, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = dropout
        self.num_layers = num_layers

        # # Initial normalization for embedding
        # self.embed_norm = nn.LayerNorm(emb_dim)
        
        layers = []
        norms = [] 
        input_dim = emb_dim
        
        for i in range(num_layers):
            # Pre-normalization before each conv layer
            norms.append(nn.LayerNorm(input_dim))
            
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            layers.append(WeightedGINConv(mlp))
            input_dim = hidden_dim
        
        self.convs = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)

        self.pool = global_mean_pool
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, data):
        x_idx = data.x.view(-1)
        x = self.embedding(x_idx)
        # x = self.embed_norm(x)  # Normalize initial embeddings

        edge_index = data.edge_index
        edge_attr = data.edge_attr
        if edge_attr is not None:
            edge_attr = edge_attr.view(-1)

        # Convolutional layers with PRE-normalization
        for i, conv in enumerate(self.convs):
            # Apply LayerNorm BEFORE the convolution
            x = self.norms[i](x)
            x = conv(x, edge_index, edge_attr=edge_attr)
            # x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        g = self.pool(x, batch)
        out = self.mlp(g)
        return F.softmax(out, dim=1)