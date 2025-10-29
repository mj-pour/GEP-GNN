import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, to_dense_batch
import random
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from collections import Counter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, roc_auc_score
import time
    

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
    

def train(
        graphs,
        model,
        batch_size=64,
        epoch_n=50,
        learning_rate=1e-3,
        weighted_sampling=False,
        use_scheduler=True,
        scheduler_patience=10,
        scheduler_factor=0.5,
        use_gradient_clipping=True,
        clip_value=1.0,
        model_name="model.pt",
        device="cuda" if torch.cuda.is_available() else "cpu",
        val_split=0.2,
        random_seed=111,
):
    
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # === Device setup ===
    model = model.to(device)

    # === Dataset Split ===
    data_indices = list(range(len(graphs)))
    test_indices = random.sample(data_indices, int(len(graphs) * val_split))
    trainset = [graphs[i] for i in data_indices if i not in test_indices]
    testset = [graphs[i] for i in data_indices if i in test_indices]

    # ========== Weighted sampling for class imbalance ==========
    if weighted_sampling:
        label_count = Counter([int(data.y) for data in graphs])
        weights = [1.0 / label_count[int(data.y)] for data in trainset]
        sampler = WeightedRandomSampler(weights, num_samples=len(trainset), replacement=True)
        train_loader = DataLoader(trainset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    print(f"Train size: {len(trainset)}, Validation size: {len(testset)}")

    # === Optimizer and Scheduler ===
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = None
    if use_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=scheduler_patience, factor=scheduler_factor)

    # === Loss Function ===
    criterion = torch.nn.CrossEntropyLoss()

    # === Logging Containers ===
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    print(f"\n Starting training for {epoch_n} epochs on {device.upper()}...\n")

    for epoch in range(1, epoch_n + 1):
        t0 = time.time()
        model.train()
        total_loss = 0
        y_true, y_pred = [], []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()

            # Optional gradient clipping
            if use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()
            total_loss += loss.item()

            preds = out.argmax(dim=1).detach().cpu().numpy()
            labels = batch.y.detach().cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels)

        avg_train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(y_true, y_pred)
        train_losses.append(avg_train_loss)

        # === Validation Phase ===
        val_loss, val_acc, val_auc = np.nan, np.nan, np.nan
        if val_loader:
            model.eval()
            y_true_val, y_pred_val, y_prob_val = [], [], []
            with torch.no_grad():
                total_val_loss = 0
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    loss = criterion(out, batch.y)
                    total_val_loss += loss.item()

                    preds = out.argmax(dim=1).cpu().numpy()
                    probs = out[:, 1].cpu().numpy()  # probability of class 1
                    labels = batch.y.cpu().numpy()

                    y_true_val.extend(labels)
                    y_pred_val.extend(preds)
                    y_prob_val.extend(probs)

            val_loss = total_val_loss / len(val_loader)
            val_acc = accuracy_score(y_true_val, y_pred_val)
            val_auc = roc_auc_score(y_true_val, y_prob_val) if len(set(y_true_val)) > 1 else np.nan
            val_losses.append(val_loss)

            if use_scheduler and scheduler is not None:
                scheduler.step(val_loss)

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_name)

        dt = time.time() - t0

        print(f"Epoch [{epoch:03d}/{epoch_n}] "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.3f} "
              f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | Val AUC: {val_auc:.3f} "
              f"| Time: {dt:.1f}s")

    print(f"\n✅ Training completed. Best model saved as: {model_name}")
    return train_losses, val_losses
