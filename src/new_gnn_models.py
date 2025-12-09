import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


class GraphSAGE(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        pool="mean"
    ):
        super().__init__()

        # ----- Embedding -----
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        # ----- GraphSAGE layers -----
        self.convs = nn.ModuleList()
        self.dropout = dropout

        in_dim = emb_dim
        for i in range(num_layers):
            self.convs.append(SAGEConv(in_dim, hidden_dim))
            in_dim = hidden_dim

        # ----- Global pooling -----
        if pool == "mean":
            self.pool = global_mean_pool
        elif pool == "max":
            self.pool = global_max_pool
        elif pool == "sum":
            self.pool = global_add_pool
        else:
            raise ValueError("pool must be 'mean', 'max', or 'sum'.")

        # ----- Final MLP -----
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)   # binary classification
        )

    def forward(self, data):
        # ---- Node embedding ----
        x_idx = data.x.view(-1)
        x = self.embedding(x_idx)

        edge_index = data.edge_index

        # ---- GraphSAGE layers (NO edge weights allowed) ----
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)

            if i < len(self.convs) - 1 and self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # ---- Global pooling ----
        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        g = self.pool(x, batch)

        # ---- Classifier ----
        out = self.mlp(g)
        return out

class WeightedGCN(nn.Module):
    def __init__(
            self, 
            vocab_size, 
            emb_dim=128, 
            hidden_dim=128, 
            num_layers=2, 
            dropout=0.2,
            pool="mean"
    ):
        super().__init__()

        # ----- Embedding -----
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        # ----- GCN layers -----
        self.convs = nn.ModuleList()
        self.dropout = dropout
        
        in_dim = emb_dim
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim, hidden_dim))
            in_dim = hidden_dim
            
        # ----- Global pooling -----
        if pool == "mean":
            self.pool = global_mean_pool
        elif pool == "max":
            self.pool = global_max_pool
        elif pool == "sum":
            self.pool = global_add_pool
        else:
            raise ValueError("pool must be 'mean', 'max', or 'sum'.")
        
        # ----- Final MLP -----
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 2)
        )

    def forward(self, data, edge_weight=None):
        # ---- Node embedding ----
        x_idx = data.x.view(-1)
        x = self.embedding(x_idx)

        edge_index = data.edge_index

        if edge_weight is None and hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_weight = data.edge_attr.view(-1)

        # ---- GCN layers        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            if i < len(self.convs) - 1 and self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # ---- Global pooling ----
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        g = self.pool(x, batch)
        
        # ---- Classifier ----        
        out = self.mlp(g)
        return out
    
class DefaultGINModel(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        emb_dim=128, 
        hidden_dim=128, 
        num_layers=2, 
        dropout=0.2,
        pool="mean"
    ):
        super(DefaultGINModel, self).__init__()

        # ----- Embedding -----
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        # ----- GIN layers -----
        self.dropout = dropout

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

        # ----- Global pooling -----
        if pool == "mean":
            self.pool = global_mean_pool
        elif pool == "max":
            self.pool = global_max_pool
        elif pool == "sum":
            self.pool = global_add_pool
        else:
            raise ValueError("pool must be 'mean', 'max', or 'sum'.")
        
        # ----- Final MLP -----        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, data):
        # ---- Node embedding ----
        x_idx = data.x.view(-1)
        x = self.embedding(x_idx)
        edge_index = data.edge_index

        # ---- GIN layers  
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            if i < len(self.convs) - 1 and self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # ---- Global pooling ----
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        g = self.pool(x, batch)

        # ---- Classifier ----  
        out = self.mlp(g)
        return out