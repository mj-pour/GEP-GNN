import torch
import random
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from collections import Counter
from sklearn.metrics import roc_auc_score, confusion_matrix


# ---------- Helper metric functions ----------
def calculate_metrics(y_true, y_pred):
    """Return TP, FN, FP, TN, Sensitivity, Specificity, Accuracy, AUC"""
    TP, FN, FP, TN = 0, 0, 0, 0
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    if cm.shape == (2, 2):
        TP, FN, FP, TN = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    else:
        # fallback if only one class in batch
        if y_true[0] == 1:
            TP = len(y_true)
        else:
            TN = len(y_true)
    SN = TP / (TP + FN + 1e-6)
    SP = TN / (TN + FP + 1e-6)
    ACC = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    try:
        AUC = roc_auc_score(y_true, y_pred)
    except Exception:
        AUC = 0.0
    return TP, FN, FP, TN, SN, SP, ACC, AUC


# ---------- Training ----------
def train(dataset, model, learning_rate=1e-4, batch_size=64, epoch_n=100,
          random_seed=111, val_split=0.3, weighted_sampling=True,
          model_name="model.pt",
          device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    random.seed(random_seed)
    torch.manual_seed(random_seed)

    data_indices = list(range(len(dataset)))
    test_indices = random.sample(data_indices, int(len(dataset) * val_split))
    trainset = [dataset[i] for i in data_indices if i not in test_indices]
    testset = [dataset[i] for i in data_indices if i in test_indices]

    # Weighted sampling for class imbalance
    if weighted_sampling:
        label_count = Counter([int(data.y) for data in dataset])
        weights = [1.0 / label_count[int(data.y)] for data in trainset]
        sampler = WeightedRandomSampler(weights, num_samples=len(trainset), replacement=True)
        train_loader = DataLoader(trainset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    print(f"Train size: {len(trainset)}, Validation size: {len(testset)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, train_accuracies, val_accuracies = [], [], []
    best_val_acc = 0

    for epoch in range(epoch_n):
        model.train()
        total_loss = 0.0
        total_acc = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            edge_weight = batch.edge_attr.squeeze() if hasattr(batch, 'edge_attr') else None

            pred = model(batch, edge_weight=edge_weight) if 'edge_weight' in model.forward.__code__.co_varnames else model(batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += (pred.argmax(dim=1) == batch.y).float().mean().item()

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        val_acc = evaluate(val_loader, model, device)
        train_losses.append(avg_loss)
        train_accuracies.append(avg_acc)
        val_accuracies.append(val_acc)

        if val_acc > best_val_acc:
            torch.save(model, model_name)
            best_val_acc = val_acc

        print(f"Epoch [{epoch+1}/{epoch_n}] Loss: {avg_loss:.4f} | "
              f"Train Acc: {avg_acc:.4f} | Val Acc: {val_acc:.4f}")

    # Plot learning curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curves")
    plt.tight_layout()
    plt.show()

    return model


# ---------- Evaluation ----------
def evaluate(loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            edge_weight = batch.edge_attr.squeeze() if hasattr(batch, 'edge_attr') else None
            pred = model(batch, edge_weight=edge_weight) if 'edge_weight' in model.forward.__code__.co_varnames else model(batch)
            correct += (pred.argmax(dim=1) == batch.y).sum().item()
            total += batch.num_graphs
    return correct / total


# ---------- Cross-species Test ----------
def test(test_dataset, model_name="model.pt",
         device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    model = torch.load(model_name, map_location=device)
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            edge_weight = batch.edge_attr.squeeze() if hasattr(batch, 'edge_attr') else None
            pred = model(batch, edge_weight=edge_weight) if 'edge_weight' in model.forward.__code__.co_varnames else model(batch)
            all_preds.extend(pred.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    TP, FN, FP, TN, SN, SP, ACC, AUC = calculate_metrics(all_labels, all_preds)
    print(f"TP={TP}, FN={FN}, FP={FP}, TN={TN}")
    print(f"SN={SN:.3f}, SP={SP:.3f}, ACC={ACC:.3f}, AUC={AUC:.3f}")

    return {"TP": TP, "FN": FN, "FP": FP, "TN": TN, "SN": SN, "SP": SP, "ACC": ACC, "AUC": AUC}