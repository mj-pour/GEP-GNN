# import torch
# import random
# import numpy as np
# import time
# from torch.utils.data import WeightedRandomSampler
# from torch_geometric.loader import DataLoader
# from torch.optim import Adam
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from sklearn.metrics import accuracy_score, roc_auc_score
# from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
# from collections import Counter
# import matplotlib.pyplot as plt

# class EarlyStopping:
#     def __init__(self, patience=20, min_delta=0.0, verbose=True, save_path="best_model.pt"):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.verbose = verbose
#         self.save_path = save_path

#         self.best_loss = float('inf')
#         self.counter = 0
#         self.should_stop = False

#     def step(self, val_loss, model):
#         if val_loss < self.best_loss - self.min_delta:
#             self.best_loss = val_loss
#             self.counter = 0
#             torch.save(model.state_dict(), self.save_path)
#             if self.verbose:
#                 print(f"  → EarlyStopping: Validation loss improved to {val_loss:.4f}, saving model.")
#         else:
#             self.counter += 1
#             if self.verbose:
#                 print(f"  → EarlyStopping: No improvement ({self.counter}/{self.patience})")

#             if self.counter >= self.patience:
#                 self.should_stop = True


# def train(
#         graphs,
#         model,
#         batch_size=64,
#         epoch_n=50,
#         learning_rate=1e-3,
#         weighted_sampling=True,
#         use_scheduler=True,
#         scheduler_patience=10,
#         scheduler_factor=0.5,
#         earlystop_patience=20,
#         use_gradient_clipping=True,
#         clip_value=1.0,
#         lambda_aux = 1e-3,
#         model_name="model.pt",
#         device="cuda" if torch.cuda.is_available() else "cpu",
#         val_split=0.2,
#         random_seed=111,
# ):
#     random.seed(random_seed)
#     torch.manual_seed(random_seed)

#     # === Device setup ===
#     model = model.to(device)

#     # === Dataset Split ===
#     data_indices = list(range(len(graphs)))
#     test_indices = random.sample(data_indices, int(len(graphs) * val_split))
#     trainset = [graphs[i] for i in data_indices if i not in test_indices]
#     valset = [graphs[i] for i in data_indices if i in test_indices]

#     # === Weighted Sampling (for class imbalance) ===
#     if weighted_sampling:
#         label_count = Counter([int(data.y) for data in graphs])
#         weights = [1.0 / label_count[int(data.y)] for data in trainset]
#         sampler = WeightedRandomSampler(weights, num_samples=len(trainset), replacement=True)
#         train_loader = DataLoader(trainset, batch_size=batch_size, sampler=sampler)
#     else:
#         train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

#     val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

#     print(f"Train size: {len(trainset)}, Validation size: {len(valset)}")

#     # === Optimizer and Scheduler ===
#     optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
#     scheduler = None
#     if use_scheduler:
#         scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=scheduler_patience, factor=scheduler_factor)

#     # === Loss Function ===
#     criterion = torch.nn.CrossEntropyLoss()

    
#     # ---- EarlyStopping ----
#     early_stopper = EarlyStopping(
#         patience=earlystop_patience,
#         save_path=model_name,
#         verbose=True
#     )

#     print(f"Train={len(trainset)}, Val={len(valset)}")

#     # === Logging Containers ===
#     train_losses, val_losses = [], []
#     train_aux_losses = []
#     train_aucs, val_aucs = [], []
#     train_accuracies, val_accuracies = [], []

#     print(f"\n🚀 Starting training for {epoch_n} epochs on {device.upper()}...\n")

#     for epoch in range(1, epoch_n + 1):
#         t0 = time.time()
#         model.train()
#         total_loss = 0
#         total_aux_loss = 0
#         y_true, y_pred, y_prob = [], [], []

#         lambda_aux = lambda_aux  # Hyperparameter -> weigh the auxiliary loss (for tuning stability)
#         for batch in train_loader:
#             batch = batch.to(device)
#             optimizer.zero_grad()
#             out, aux_loss = model(batch)
#             ce_loss = criterion(out, batch.y)
#             loss = ce_loss + lambda_aux * aux_loss # combine main and auxiliary losses
#             loss.backward()

#             # Optional gradient clipping
#             if use_gradient_clipping:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

#             optimizer.step()
#             total_loss += loss.item()
#             total_aux_loss += aux_loss.item()

#             preds = out.argmax(dim=1).detach().cpu().numpy()
#             probs = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
#             labels = batch.y.detach().cpu().numpy()

#             y_pred.extend(preds)
#             y_prob.extend(probs)
#             y_true.extend(labels)

#         avg_train_loss = total_loss / len(train_loader)
#         avg_train_aux_loss = total_aux_loss / len(train_loader)
#         train_acc = accuracy_score(y_true, y_pred)
#         train_auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else np.nan
#         train_losses.append(avg_train_loss)
#         train_aux_losses.append(avg_train_aux_loss)
#         train_accuracies.append(train_acc)
#         train_aucs.append(train_auc)

#         # === Validation Phase ===
#         model.eval()
#         y_true_val, y_pred_val, y_prob_val = [], [], []
#         with torch.no_grad():
#             total_val_loss = 0
#             for batch in val_loader:
#                 batch = batch.to(device)
#                 out, _ = model(batch)
#                 loss = criterion(out, batch.y)
#                 total_val_loss += loss.item()

#                 preds = out.argmax(dim=1).cpu().numpy()
#                 probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
#                 labels = batch.y.cpu().numpy()

#                 y_true_val.extend(labels)
#                 y_pred_val.extend(preds)
#                 y_prob_val.extend(probs)

#         val_loss = total_val_loss / len(val_loader)
#         val_acc = accuracy_score(y_true_val, y_pred_val)
#         val_auc = roc_auc_score(y_true_val, y_prob_val) if len(set(y_true_val)) > 1 else np.nan
#         val_losses.append(val_loss)
#         val_accuracies.append(val_acc)
#         val_aucs.append(val_auc)

#         if use_scheduler and scheduler is not None:
#             scheduler.step(val_loss)

#         dt = time.time() - t0
#         print(f"Epoch [{epoch:03d}/{epoch_n}] "
#               f"Train CE Loss: {avg_train_loss:.4f} | Train AUX Loss: {avg_train_aux_loss:.4f} | Train AUC: {train_auc:.3f} | Train Acc: {train_acc:.3f} "
#               f"| Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.3f} | Val Acc: {val_acc:.3f} "
#               f"| Time: {dt:.1f}s")
        
#         # ---- Early stopping ----
#         early_stopper.step(val_loss, model)
#         if early_stopper.should_stop:
#             print("\n⛔ Early stopping triggered!")
#             break

#     print(f"\n✅ Training completed. Best model saved as: {model_name}")

#     # === Plot Learning Curves ===
#     plt.figure(figsize=(12, 5))

#     # --- Loss Curves ---
#     plt.subplot(1, 2, 1)
#     plt.plot(train_losses, label="Train CE Loss", linewidth=2)
#     plt.plot(train_aux_losses, label="Train AUX Loss", linewidth=2)
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.title("Loss Curve")

#     # --- AUC Curves ---
#     plt.subplot(1, 2, 2)
#     plt.plot(train_aucs, label="Train AUC", linewidth=2)
#     plt.plot(val_aucs, label="Validation AUC", linewidth=2)
#     plt.xlabel("Epoch")
#     plt.ylabel("AUC")
#     plt.legend()
#     plt.title("AUC per Epoch")

#     plt.tight_layout()
#     plt.show()

#     return model, train_losses, val_losses, train_aucs, val_aucs

# def test(
#         model,
#         test_graphs,
#         model_path,
#         batch_size=64,
#         device="cuda" if torch.cuda.is_available() else "cpu",
#         return_predictions=False
# ):

#     # --- Load model and prepare ---
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model = model.to(device)
#     model.eval()

#     # --- Data Loader ---
#     test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
#     criterion = torch.nn.CrossEntropyLoss()

#     # --- Logging containers ---
#     total_test_loss = 0
#     y_true, y_pred, y_prob = [], [], []

#     print(f"Testing on {len(test_graphs)} samples...")

#     # --- Evaluation loop ---
#     with torch.no_grad():
#         for batch in test_loader:
#             batch = batch.to(device)
#             out, (_, _) = model(batch)
#             loss = criterion(out, batch.y)
#             total_test_loss += loss.item()

#             preds = out.argmax(dim=1).cpu().numpy()
#             probs = out[:, 1].cpu().numpy() if out.size(1) > 1 else np.zeros_like(preds)
#             labels = batch.y.cpu().numpy()

#             y_true.extend(labels)
#             y_pred.extend(preds)
#             y_prob.extend(probs)

#     # --- Compute metrics ---
#     test_loss = total_test_loss / len(test_loader)
#     test_acc = accuracy_score(y_true, y_pred)
#     test_auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else np.nan
#     test_f1 = f1_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)
#     cm = confusion_matrix(y_true, y_pred)
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel().tolist()
#     SN = tp / (tp + fn)
#     SP = tn / (tn + fp)

#     # --- Display summary ---
#     print("\n=== Final Test Results ===")
#     print(f"Test Loss: {test_loss:.4f}")
#     print("Confusion Matrix:\n", cm)
#     print(f"\nTrue Negative: {tn}")
#     print(f"False Positive: {fp}")
#     print(f"False Negative: {fn}")
#     print(f"True Positive: {tp}")
#     print(f"Sensitivity: {SN}")
#     print(f"Specificity: {SP}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall:    {recall:.4f}")
#     print(f"Accuracy:  {test_acc:.4f}")
#     print(f"AUC:       {test_auc:.4f}")
#     print(f"F1 Score:  {test_f1:.4f}")


#     # Prepare results
#     results = {
#         'test_loss': test_loss,
#         'test_accuracy': test_acc,
#         'test_auc': test_auc,
#         'test_f1': test_f1,
#         'test_precision': precision,
#         'test_recall': recall,
#         'TN': tn,
#         'FP': fp,
#         'FN': fn,
#         'TP': tp
#     }
    
#     if return_predictions:
#         results['true_labels'] = y_true
#         results['predictions'] = y_pred
#         results['probabilities'] = y_prob

#     return 



# import os
# import time
# import random
# import numpy as np
# from collections import Counter
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score
# import torch
# from torch.utils.data import WeightedRandomSampler
# from torch_geometric.loader import DataLoader
# from torch.optim import Adam
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import matplotlib.pyplot as plt

# class EarlyStop:
#     def __init__(self, patience=20, min_delta=0.0, verbose=True, save_path="best.pt"):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.verbose = verbose
#         self.save_path = save_path

#         self.best_loss = float('inf')
#         self.counter = 0
#         self.should_stop = False

#     def step(self, val_loss, model):
#         if val_loss < self.best_loss - self.min_delta:
#             self.best_loss = val_loss
#             self.counter = 0
#             torch.save(model.state_dict(), self.save_path)
#             if self.verbose:
#                 print(f"  → EarlyStop: val_loss improved to {val_loss:.4f}; saved {self.save_path}")
#         else:
#             self.counter += 1
#             if self.verbose:
#                 print(f"  → EarlyStop: no improvement ({self.counter}/{self.patience})")
#             if self.counter >= self.patience:
#                 self.should_stop = True

# def train_one_fold(graphs,
#                    model,
#                    device="cuda",
#                    batch_size=32,
#                    epoch_n=100,
#                    learning_rate=1e-3,
#                    lambda_aux=0.1,
#                    weighted_sampling=True,
#                    use_scheduler=True,
#                    scheduler_patience=10,
#                    scheduler_factor=0.5,
#                    use_gradient_clipping=True,
#                    clip_value=1.0,
#                    earlystop_patience=20,
#                    model_path="best_model.pt",
#                    val_split=0.2,
#                    random_seed=111):
#     """
#     Train/validate single run (internal splitting). Returns model (best weights loaded) and metrics history.
#     """
#     random.seed(random_seed)
#     torch.manual_seed(random_seed)
#     np.random.seed(random_seed)

#     device = torch.device(device if torch.cuda.is_available() else "cpu")
#     model = model.to(device)

#     # split
#     idx = list(range(len(graphs)))
#     random.shuffle(idx)
#     n_val = int(len(idx) * val_split)
#     val_idx = idx[:n_val]
#     train_idx = idx[n_val:]
#     trainset = [graphs[i] for i in train_idx]
#     valset = [graphs[i] for i in val_idx]

#     # loaders
#     if weighted_sampling:
#         label_count = Counter([int(g.y) for g in trainset])
#         weights = [1.0 / label_count[int(g.y)] for g in trainset]
#         sampler = WeightedRandomSampler(weights, num_samples=len(trainset), replacement=True)
#         train_loader = DataLoader(trainset, batch_size=batch_size, sampler=sampler)
#     else:
#         train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

#     val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

#     optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=5e-5)
#     scheduler = None
#     if use_scheduler:
#         scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=scheduler_patience, factor=scheduler_factor)

#     criterion = torch.nn.CrossEntropyLoss()
#     earlystop = EarlyStop(patience=earlystop_patience, save_path=model_path, verbose=True)

#     # logs
#     train_losses, val_losses = [], []
#     train_aucs, val_aucs = [], []
#     train_aux_losses = []

#     best_epoch = 0

#     print(f"Train={len(trainset)}, Val={len(valset)}, batch={batch_size}, device={device}")

#     for epoch in range(1, epoch_n + 1):
#         t0 = time.time()
#         model.train()
#         total_loss, total_aux = 0.0, 0.0
#         y_true_train, y_prob_train = [], []

#         for batch in train_loader:
#             batch = batch.to(device)
#             optimizer.zero_grad()
#             logits, aux = model(batch)
#             ce = criterion(logits, batch.y)
#             loss = ce + lambda_aux * aux
#             loss.backward()
#             if use_gradient_clipping:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
#             optimizer.step()

#             total_loss += loss.item()
#             total_aux += aux.item()

#             probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
#             y_prob_train.extend(probs)
#             y_true_train.extend(batch.y.detach().cpu().numpy())

#         avg_train_loss = total_loss / len(train_loader)
#         avg_train_aux = total_aux / len(train_loader)
#         train_auc = roc_auc_score(y_true_train, y_prob_train) if len(set(y_true_train)) > 1 else np.nan

#         # validation
#         model.eval()
#         total_val_loss = 0.0
#         y_true_val, y_prob_val = [], []

#         with torch.no_grad():
#             for batch in val_loader:
#                 batch = batch.to(device)
#                 logits, aux = model(batch)
#                 ce = criterion(logits, batch.y)
#                 total_val_loss += (ce + lambda_aux * aux).item()

#                 probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
#                 y_prob_val.extend(probs)
#                 y_true_val.extend(batch.y.cpu().numpy())

#         avg_val_loss = total_val_loss / len(val_loader)
#         val_auc = roc_auc_score(y_true_val, y_prob_val) if len(set(y_true_val)) > 1 else np.nan

#         train_losses.append(avg_train_loss)
#         train_aux_losses.append(avg_train_aux)
#         val_losses.append(avg_val_loss)
#         train_aucs.append(train_auc)
#         val_aucs.append(val_auc)

#         if scheduler is not None:
#             scheduler.step(avg_val_loss)

#         print(f"Epoch [{epoch}/{epoch_n}] TrainLoss={avg_train_loss:.4f} | TrainAux={avg_train_aux:.4f} | TrainAUC={train_auc:.3f} "
#               f"| ValLoss={avg_val_loss:.4f} | ValAUC={val_auc:.3f} | Time={time.time()-t0:.1f}s")

#         # early stop
#         earlystop.step(avg_val_loss, model)
#         if earlystop.should_stop:
#             print("Early stopping triggered.")
#             break

#         best_epoch = epoch

#     # load best model
#     if os.path.exists(model_path):
#         model.load_state_dict(torch.load(model_path, map_location=device))
#     return {
#         "model": model,
#         "train_losses": train_losses,
#         "val_losses": val_losses,
#         "train_aucs": train_aucs,
#         "val_aucs": val_aucs,
#         "train_aux_losses": train_aux_losses,
#         "best_epoch": best_epoch,
#         "model_path": model_path
#     }


# def k_fold_train(graphs,
#                  model_class,
#                  model_kwargs,
#                  k=5,
#                  seed=111,
#                  **train_kwargs):
#     """
#     Stratified k-fold training wrapper.
#     - graphs: list of Data objects (with .y)
#     - model_class: class (not instance); we'll instantiate per fold
#     - model_kwargs: kwargs to pass to model_class(...)
#     - train_kwargs: forwarded to train_one_fold
#     """
#     labels = np.array([int(g.y) for g in graphs])
#     skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

#     fold_results = []
#     for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
#         print("\n" + "="*60)
#         print(f"FOLD {fold}/{k}")
#         print("="*60)

#         # instantiate a fresh model
#         model = model_class(**model_kwargs)

#         model_path = train_kwargs.get("model_path", f"best_model_fold{fold}.pt")
#         # ensure unique path per fold
#         train_kwargs_fold = train_kwargs.copy()
#         train_kwargs_fold["model_path"] = model_path
#         train_kwargs_fold["val_split"] = len(val_idx)/len(labels)  # pass a matching val split if needed

#         # combine training graphsets for train_one_fold (it internally splits)
#         result = train_one_fold(graphs, model, random_seed=seed+fold, **train_kwargs_fold)
#         fold_results.append({
#             "fold": fold,
#             "model_path": result["model_path"],
#             "best_epoch": result["best_epoch"],
#             "train_losses": result["train_losses"],
#             "val_losses": result["val_losses"],
#             "train_aucs": result["train_aucs"],
#             "val_aucs": result["val_aucs"]
#         })

#     return fold_results







import torch
import random
import numpy as np
import time
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from collections import Counter
import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0, verbose=True, save_path="best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.save_path = save_path

        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            if self.verbose:
                print(f"  → EarlyStopping: Validation loss improved to {val_loss:.4f}, saving model.")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  → EarlyStopping: No improvement ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.should_stop = True


def train(
        graphs,
        model,
        batch_size=64,
        epoch_n=50,
        learning_rate=1e-3,
        weighted_sampling=True,
        use_scheduler=True,
        scheduler_patience=10,
        scheduler_factor=0.5,
        earlystop_patience=20,
        use_gradient_clipping=True,
        clip_value=1.0,
        lambda_aux=1e-3,
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
    valset = [graphs[i] for i in data_indices if i in test_indices]

    # === Weighted Sampling (for class imbalance) ===
    if weighted_sampling:
        label_count = Counter([int(data.y) for data in graphs])
        weights = [1.0 / label_count[int(data.y)] for data in trainset]
        sampler = WeightedRandomSampler(weights, num_samples=len(trainset), replacement=True)
        train_loader = DataLoader(trainset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    print(f"Train size: {len(trainset)}, Validation size: {len(valset)}")

    # === Optimizer and Scheduler ===
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = None
    if use_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=scheduler_patience, factor=scheduler_factor)

    # === Loss Function ===
    criterion = torch.nn.CrossEntropyLoss()

    
    # ---- EarlyStopping ----
    early_stopper = EarlyStopping(
        patience=earlystop_patience,
        save_path=model_name,
        verbose=True
    )

    print(f"Train={len(trainset)}, Val={len(valset)}")

    # === Logging Containers ===
    train_losses, val_losses = [], []
    train_aucs, val_aucs = [], []
    train_accuracies, val_accuracies = [], []

    print(f"\n🚀 Starting training for {epoch_n} epochs on {device.upper()}...\n")

    for epoch in range(1, epoch_n + 1):
        t0 = time.time()
        model.train()
        total_loss = 0
        y_true, y_pred, y_prob = [], [], []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out, aux_loss = model(batch)  # aux_loss is now 0.0 from the model
            loss = criterion(out, batch.y)  # Only use cross-entropy loss
            loss.backward()

            # Optional gradient clipping
            if use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()
            total_loss += loss.item()

            preds = out.argmax(dim=1).detach().cpu().numpy()
            probs = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
            labels = batch.y.detach().cpu().numpy()

            y_pred.extend(preds)
            y_prob.extend(probs)
            y_true.extend(labels)

        avg_train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(y_true, y_pred)
        train_auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else np.nan
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        train_aucs.append(train_auc)

        # === Validation Phase ===
        model.eval()
        y_true_val, y_pred_val, y_prob_val = [], [], []
        with torch.no_grad():
            total_val_loss = 0
            for batch in val_loader:
                batch = batch.to(device)
                out, _ = model(batch)
                loss = criterion(out, batch.y)
                total_val_loss += loss.item()

                preds = out.argmax(dim=1).cpu().numpy()
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                labels = batch.y.cpu().numpy()

                y_true_val.extend(labels)
                y_pred_val.extend(preds)
                y_prob_val.extend(probs)

        val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(y_true_val, y_pred_val)
        val_auc = roc_auc_score(y_true_val, y_prob_val) if len(set(y_true_val)) > 1 else np.nan
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_aucs.append(val_auc)

        if use_scheduler and scheduler is not None:
            scheduler.step(val_loss)

        dt = time.time() - t0
        print(f"Epoch [{epoch:03d}/{epoch_n}] "
              f"Train Loss: {avg_train_loss:.4f} | Train AUC: {train_auc:.3f} | Train Acc: {train_acc:.3f} "
              f"| Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.3f} | Val Acc: {val_acc:.3f} "
              f"| Time: {dt:.1f}s")
        
        # ---- Early stopping ----
        early_stopper.step(val_loss, model)
        if early_stopper.should_stop:
            print("\n⛔ Early stopping triggered!")
            break

    print(f"\n✅ Training completed. Best model saved as: {model_name}")

    # === Plot Learning Curves ===
    plt.figure(figsize=(12, 5))

    # --- Loss Curves ---
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    # --- AUC Curves ---
    plt.subplot(1, 2, 2)
    plt.plot(train_aucs, label="Train AUC", linewidth=2)
    plt.plot(val_aucs, label="Validation AUC", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.title("AUC per Epoch")

    plt.tight_layout()
    plt.show()

    return model, train_losses, val_losses, train_aucs, val_aucs

def test(
        model,
        test_graphs,
        model_path,
        batch_size=64,
        device="cuda" if torch.cuda.is_available() else "cpu",
        return_predictions=False
):

    # --- Load model and prepare ---
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # --- Data Loader ---
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()

    # --- Logging containers ---
    total_test_loss = 0
    y_true, y_pred, y_prob = [], [], []

    print(f"Testing on {len(test_graphs)} samples...")

    # --- Evaluation loop ---
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out, _ = model(batch)  # Ignore the second return value (aux_loss)
            loss = criterion(out, batch.y)
            total_test_loss += loss.item()

            preds = out.argmax(dim=1).cpu().numpy()
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy() if out.size(1) > 1 else np.zeros_like(preds)
            labels = batch.y.cpu().numpy()

            y_true.extend(labels)
            y_pred.extend(preds)
            y_prob.extend(probs)

    # --- Compute metrics ---
    test_loss = total_test_loss / len(test_loader)
    test_acc = accuracy_score(y_true, y_pred)
    test_auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else np.nan
    test_f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel().tolist()
    SN = tp / (tp + fn)
    SP = tn / (tn + fp)

    # --- Display summary ---
    print("\n=== Final Test Results ===")
    print(f"Test Loss: {test_loss:.4f}")
    print("Confusion Matrix:\n", cm)
    print(f"\nTrue Negative: {tn}")
    print(f"False Positive: {fp}")
    print(f"False Negative: {fn}")
    print(f"True Positive: {tp}")
    print(f"Sensitivity: {SN}")
    print(f"Specificity: {SP}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"AUC:       {test_auc:.4f}")
    print(f"F1 Score:  {test_f1:.4f}")

    # Prepare results
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_auc': test_auc,
        'test_f1': test_f1,
        'test_precision': precision,
        'test_recall': recall,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp
    }
    
    if return_predictions:
        results['true_labels'] = y_true
        results['predictions'] = y_pred
        results['probabilities'] = y_prob

    return results