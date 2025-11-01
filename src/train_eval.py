# import torch
# import random
# import matplotlib.pyplot as plt
# from torch import nn, optim
# from torch.utils.data import WeightedRandomSampler
# from torch_geometric.loader import DataLoader
# from collections import Counter
# from sklearn.metrics import roc_auc_score, confusion_matrix


# # ---------- Helper metric functions ----------
# def calculate_metrics(y_true, y_pred):
#     """Return TP, FN, FP, TN, Sensitivity, Specificity, Accuracy, AUC"""
#     TP, FN, FP, TN = 0, 0, 0, 0
#     cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
#     if cm.shape == (2, 2):
#         TP, FN, FP, TN = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
#     else:
#         # fallback if only one class in batch
#         if y_true[0] == 1:
#             TP = len(y_true)
#         else:
#             TN = len(y_true)
#     SN = TP / (TP + FN + 1e-6)
#     SP = TN / (TN + FP + 1e-6)
#     ACC = (TP + TN) / (TP + TN + FP + FN + 1e-6)
#     try:
#         AUC = roc_auc_score(y_true, y_pred)
#     except Exception:
#         AUC = 0.0
#     return TP, FN, FP, TN, SN, SP, ACC, AUC


# # ---------- Training ----------
# def train(dataset, model, learning_rate=1e-4, batch_size=64, epoch_n=100,
#           random_seed=111, val_split=0.3, weighted_sampling=True,
#           model_name="model.pt",
#           device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

#     random.seed(random_seed)
#     torch.manual_seed(random_seed)

#     data_indices = list(range(len(dataset)))
#     test_indices = random.sample(data_indices, int(len(dataset) * val_split))
#     trainset = [dataset[i] for i in data_indices if i not in test_indices]
#     testset = [dataset[i] for i in data_indices if i in test_indices]

#     # Weighted sampling for class imbalance
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

#     train_losses, train_accuracies, val_accuracies = [], [], []
#     best_val_acc = 0

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
#             optimizer.step()

#             total_loss += loss.item()
#             total_acc += (pred.argmax(dim=1) == batch.y).float().mean().item()

#         avg_loss = total_loss / len(train_loader)
#         avg_acc = total_acc / len(train_loader)
#         val_acc = evaluate(val_loader, model, device)
#         train_losses.append(avg_loss)
#         train_accuracies.append(avg_acc)
#         val_accuracies.append(val_acc)

#         if val_acc > best_val_acc:
#             torch.save(model, model_name)
#             best_val_acc = val_acc

#         print(f"Epoch [{epoch+1}/{epoch_n}] Loss: {avg_loss:.4f} | "
#               f"Train Acc: {avg_acc:.4f} | Val Acc: {val_acc:.4f}")

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


# # ---------- Cross-species Test ----------
# def test(test_dataset, model_name="model.pt",
#          device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

#     model = torch.load(model_name, map_location=device)
#     loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#     all_preds, all_labels = [], []
#     model.eval()
#     with torch.no_grad():
#         for batch in loader:
#             batch = batch.to(device)
#             edge_weight = batch.edge_attr.squeeze() if hasattr(batch, 'edge_attr') else None
#             pred = model(batch, edge_weight=edge_weight) if 'edge_weight' in model.forward.__code__.co_varnames else model(batch)
#             all_preds.extend(pred.argmax(dim=1).cpu().numpy())
#             all_labels.extend(batch.y.cpu().numpy())

#     TP, FN, FP, TN, SN, SP, ACC, AUC = calculate_metrics(all_labels, all_preds)
#     print(f"TP={TP}, FN={FN}, FP={FP}, TN={TN}")
#     print(f"SN={SN:.3f}, SP={SP:.3f}, ACC={ACC:.3f}, AUC={AUC:.3f}")

#     return {"TP": TP, "FN": FN, "FP": FP, "TN": TN, "SN": SN, "SP": SP, "ACC": ACC, "AUC": AUC}


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

    # === Logging Containers ===
    train_losses, val_losses = [], []
    train_aucs, val_aucs = [], []
    train_accuracies, val_accuracies = [], []

    best_val_loss = float('inf')

    print(f"\n🚀 Starting training for {epoch_n} epochs on {device.upper()}...\n")

    for epoch in range(1, epoch_n + 1):
        t0 = time.time()
        model.train()
        total_loss = 0
        y_true, y_pred, y_prob = [], [], []

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
                out = model(batch)
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

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_name)

        dt = time.time() - t0
        print(f"Epoch [{epoch:03d}/{epoch_n}] "
              f"Train Loss: {avg_train_loss:.4f} | Train AUC: {train_auc:.3f} | Train Acc: {train_acc:.3f} "
              f"| Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.3f} | Val Acc: {val_acc:.3f} "
              f"| Time: {dt:.1f}s")

    print(f"\n✅ Training completed. Best model saved as: {model_name}")

    # === Plot Learning Curves ===
    plt.figure(figsize=(12, 5))

    # --- Loss Curves ---
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2, linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")

    # --- AUC Curves ---
    plt.subplot(1, 2, 2)
    plt.plot(train_aucs, label="Train AUC", linewidth=2)
    plt.plot(val_aucs, label="Validation AUC", linewidth=2, linestyle='--')
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
            out = model(batch)
            loss = criterion(out, batch.y)
            total_test_loss += loss.item()

            preds = out.argmax(dim=1).cpu().numpy()
            probs = out[:, 1].cpu().numpy() if out.size(1) > 1 else np.zeros_like(preds)
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
    conf_matrix = confusion_matrix(y_true, y_pred)

    # --- Display summary ---
    print("\n=== Final Test Results ===")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"AUC:       {test_auc:.4f}")
    print(f"F1 Score:  {test_f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("Confusion Matrix:\n", conf_matrix)

    # Prepare results
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_auc': test_auc,
        'test_f1': test_f1,
        'test_precision': precision,
        'test_recall': recall,
        'confusion_matrix': conf_matrix
    }
    
    if return_predictions:
        results['true_labels'] = y_true
        results['predictions'] = y_pred
        results['probabilities'] = y_prob

    return 