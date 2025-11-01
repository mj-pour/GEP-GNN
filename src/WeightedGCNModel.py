import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool
import random
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from collections import Counter


class model(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList()
        self.dropout = dropout
        
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

    def forward(self, data, edge_weight=None):
        x_idx = data.x.view(-1)
        x = self.embedding(x_idx)
        edge_index = data.edge_index

        if edge_weight is None and hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_weight = data.edge_attr.view(-1)
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            if i < len(self.convs) - 1 and self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        g = self.pool(x, batch)
        out = self.mlp(g)
        return F.softmax(out, dim=1)
    

# def train(dataset, model, learning_rate=1e-4, batch_size=64, epoch_n=100,
#           random_seed=111, val_split=0.3, weighted_sampling=True,
#           model_name="model.pt", use_scheduler=True, use_gradient_clipping=True,
#           clip_value=1.0,  # Gradient clipping threshold
#           scheduler_patience=10,  # LR scheduler patience
#           scheduler_factor=0.5,   # LR reduction factor
#           device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

#     random.seed(random_seed)
#     torch.manual_seed(random_seed)

#     data_indices = list(range(len(dataset)))
#     test_indices = random.sample(data_indices, int(len(dataset) * val_split))
#     trainset = [dataset[i] for i in data_indices if i not in test_indices]
#     testset = [dataset[i] for i in data_indices if i in test_indices]

#     # ========== Weighted sampling for class imbalance ==========
#     if weighted_sampling:
#         label_count = Counter([int(data.y) for data in dataset])
#         weights = [1.0 / label_count[int(data.y)] for data in trainset]
#         sampler = WeightedRandomSampler(weights, num_samples=len(trainset), replacement=True)
#         train_loader = DataLoader(trainset, batch_size=batch_size, sampler=sampler)
#     else:
#         train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

#     val_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

#     print(f"Train size: {len(trainset)}, Validation size: {len(testset)}")

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
#     # ==========  LEARNING RATE SCHEDULER ==========
#     if use_scheduler:
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, 
#             mode='max',        # Monitor validation accuracy (we want it to maximize)
#             patience=scheduler_patience, 
#             factor=scheduler_factor,
#         )

#     train_losses, train_accuracies, val_accuracies = [], [], []
#     best_val_acc = 0
#     current_lr = learning_rate  # Initialize current_lr

#     for epoch in range(epoch_n):
#         model.train()
#         total_loss = 0.0
#         total_acc = 0.0

#         for batch in train_loader:
#             batch = batch.to(device)
#             optimizer.zero_grad()
#             edge_weight = batch.edge_attr.squeeze() if hasattr(batch, 'edge_attr') else None

#             pred = model(batch, edge_weight=edge_weight) if 'edge_weight' in model.forward.__code__.co_varnames else model(batch)
#             loss = criterion(pred, batch.y)
#             loss.backward()
            
#             # ========== GRADIENT CLIPPING ===================
#             if use_gradient_clipping:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
#             # ================================================
            
#             optimizer.step()

#             total_loss += loss.item()
#             total_acc += (pred.argmax(dim=1) == batch.y).float().mean().item()

#         avg_loss = total_loss / len(train_loader)
#         avg_acc = total_acc / len(train_loader)
#         val_acc = evaluate(val_loader, model, device)
        
#         # ========== UPDATE LEARNING RATE SCHEDULER ==============
#         if use_scheduler:
#             scheduler.step(val_acc)  # Pass validation accuracy to scheduler
#             current_lr = optimizer.param_groups[0]['lr'] # Update current learning rate
#         # ========================================================

#         train_losses.append(avg_loss)
#         train_accuracies.append(avg_acc)
#         val_accuracies.append(val_acc)

#         if val_acc > best_val_acc:
#             torch.save(model, model_name)
#             best_val_acc = val_acc

#         # Print current learning rate if using scheduler
#         lr_info = f" | LR: {current_lr:.2e}" if use_scheduler else ""
#         print(f"Epoch [{epoch+1}/{epoch_n}] Loss: {avg_loss:.4f} | "
#               f"Train Acc: {avg_acc:.4f} | Val Acc: {val_acc:.4f}{lr_info}")

#     # Plot learning curves
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(train_losses, label="Train Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.title("Training Loss")

#     plt.subplot(1, 2, 2)
#     plt.plot(train_accuracies, label="Train Accuracy")
#     plt.plot(val_accuracies, label="Validation Accuracy")
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.legend()
#     plt.title("Accuracy Curves")
#     plt.tight_layout()
#     plt.show()

#     return model

# # ---------- Evaluation ----------
# def evaluate(loader, model, device):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch in loader:
#             batch = batch.to(device)
#             edge_weight = batch.edge_attr.squeeze() if hasattr(batch, 'edge_attr') else None
#             pred = model(batch, edge_weight=edge_weight) if 'edge_weight' in model.forward.__code__.co_varnames else model(batch)
#             correct += (pred.argmax(dim=1) == batch.y).sum().item()
#             total += batch.num_graphs
#     return correct / total