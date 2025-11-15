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
        lambda_aux = 1e-3,
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
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=5e-5)
    scheduler = None
    if use_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=scheduler_patience, factor=scheduler_factor)

    # === Loss Function ===
    criterion = torch.nn.CrossEntropyLoss()

    # === Logging Containers ===
    train_losses, val_losses = [], []
    train_aux_losses = []
    train_aucs, val_aucs = [], []
    train_accuracies, val_accuracies = [], []

    best_val_loss = float('inf')

    print(f"\n🚀 Starting training for {epoch_n} epochs on {device.upper()}...\n")

    for epoch in range(1, epoch_n + 1):
        t0 = time.time()
        model.train()
        total_loss = 0
        total_aux_loss = 0
        y_true, y_pred, y_prob = [], [], []

        lambda_aux = lambda_aux  # Hyperparameter -> weigh the auxiliary loss (for tuning stability)
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out, aux_loss = model(batch)
            ce_loss = criterion(out, batch.y)
            loss = ce_loss + lambda_aux * aux_loss # combine main and auxiliary losses
            loss.backward()

            # Optional gradient clipping
            if use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()
            total_loss += loss.item()
            total_aux_loss += aux_loss.item()

            preds = out.argmax(dim=1).detach().cpu().numpy()
            probs = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
            labels = batch.y.detach().cpu().numpy()

            y_pred.extend(preds)
            y_prob.extend(probs)
            y_true.extend(labels)

        avg_train_loss = total_loss / len(train_loader)
        avg_train_aux_loss = total_aux_loss / len(train_loader)
        train_acc = accuracy_score(y_true, y_pred)
        train_auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else np.nan
        train_losses.append(avg_train_loss)
        train_aux_losses.append(avg_train_aux_loss)
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

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_name)

        dt = time.time() - t0
        print(f"Epoch [{epoch:03d}/{epoch_n}] "
              f"Train CE Loss: {avg_train_loss:.4f} | Train AUX Loss: {avg_train_aux_loss:.4f} | Train AUC: {train_auc:.3f} | Train Acc: {train_acc:.3f} "
              f"| Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.3f} | Val Acc: {val_acc:.3f} "
              f"| Time: {dt:.1f}s")

    print(f"\n✅ Training completed. Best model saved as: {model_name}")

    # === Plot Learning Curves ===
    plt.figure(figsize=(12, 5))

    # --- Loss Curves ---
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train CE Loss", linewidth=2)
    plt.plot(train_aux_losses, label="Train AUX Loss", linewidth=2)
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
            out, _ = model(batch)
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

    return 