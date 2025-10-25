# GEP-GNN: Graph Neural Networks for Gene Essentiality Prediction

This repository implements a modular pipeline for **gene essentiality prediction** using **Graph Neural Networks (GNNs)**.  
DNA sequences are represented as *k-mer graphs*, where edges denote transitions between k-mers, optionally weighted by transition probabilities.  
The system supports multiple GNN architectures — **Weighted GCN**, **Edge-Aware GAT**, **Graph Isomorphism Network (GIN)**, and **Hierarchical GCN (DiffPool)** — with a unified training and evaluation pipeline.

---

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Evaluation Metrics](#evaluation-metrics)
- [Example Experiment](#example-experiment)
- [Modules Overview](#modules-overview)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/YourUsername/GEP-GNN.git
cd GEP-GNN

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate     # (Windows: venv\Scripts\activate)

# Install dependencies
pip install -r requirements.txt
```

---

## Core Dependencies

torch
torch-geometric
networkx
numpy
pandas
matplotlib
scikit-learn

---

## Usage

1️ Build a Graph Dataset

Convert sequences into k-mer graphs with labels:

```python
from src.SeqEncoder import build_dataset

graphs, vocab = build_dataset(
    fasta_path="data/arabidopsis_genes.fasta",
    label_path="data/arabidopsis_labels.txt",
    k=3
)
```

2️ Train a Model

```python
from src.GNNmodel import WeightedGCNModel
from src.train_eval import train_model

model = WeightedGCNModel(vocab_size=len(vocab))
train_model(graphs, model, lr=1e-4, batch_size=32, epochs=20, model_name="models/ath-GCN.pt")
```

3️ Test on Another Species

```python
from src.train_eval import test_model
graphs2, _ = build_dataset("data/saccharomyces_genes.fasta", "data/saccharomyces_labels.txt", k=3)

test_model(graphs2, model_name="models/ath-GCN.pt")
```

---

## Model Architectures

| Model                | Description                                              | Pooling              | File          |
| -------------------- | -------------------------------------------------------- | -------------------- | ------------- |
| **WeightedGCNModel** | Graph Convolutional Network with edge weights            | Global Mean Pooling  | `GNNmodel.py` |
| **EdgeAttrGATModel** | Edge-aware Graph Attention Network                       | Global Mean Pooling  | `GNNmodel.py` |
| **GINModel**         | Graph Isomorphism Network (GINConv layers)               | Global Mean Pooling  | `GNNmodel.py` |
| **DiffPoolGCN**      | Hierarchical GCN using Differentiable Pooling (DiffPool) | Hierarchical Pooling | `GNNmodel.py` |

Each architecture is modular — you can easily add or replace layers.

---

## Evaluation Metrics

The pipeline reports both raw confusion matrix values and standard performance metrics.

Metrics:

- True Positive (TP)
- False Negative (FN)
- False Positive (FP)
- True Negative (TN)
- Sensitivity (SN) = TP / (TP + FN)
- Specificity (SP) = TN / (TN + FP)
- Accuracy (ACC) = (TP + TN) / (TP + FP + TN + FN)
- Area Under ROC Curve (AUC)

---

## Example Experiment

Example of a cross-species generalization test:

| Train Species   | Test Species    | SN   | SP   | ACC  | AUC  |
| --------------- | --------------- | ---- | ---- | ---- | ---- |
| *A. thaliana*   | *S. cerevisiae* | 0.91 | 0.78 | 0.85 | 0.80 |
| *S. cerevisiae* | *A. thaliana*   | 0.88 | 0.81 | 0.84 | 0.83 |

---

## Modules Overview

- SeqEncoder.py
  - Converts FASTA sequences into k-mer transition graphs
  - Supports:
    - Weighted edges (transition frequencies)
    - Normalized adjacency matrices
    - Node degree and sequence length features
    - Outputs PyTorch Geometric Data objects

- GNNmodel.py
  - Defines multiple architectures:
    - WeightedGCNModel – baseline GCN with weighted edges
    - EdgeAttrGATModel – edge-aware attention mechanism
    - GINModel – expressive node embedding aggregation
    - DiffPoolGCN – hierarchical pooling for graph-level embedding
    - Each architecture ends with a dense classifier head for binary classification.

- train_eval.py
  - Unified training and testing functions
  - Handles learning rate scheduling, checkpointing, and GPU detection
  - Logs loss and metric curves per epoch
- Supports cross-species evaluation

---

## Example Results Visualization

<p align="center"> <img src="docs/learning_curve_example.png" alt="Learning Curve" width="60%"> </p>

---

## Citation

If you use this repository or its methods, please cite:

> Pourmohammad, M. J., et al. (2025).  
> *Graph Neural Networks for Cross-Species Gene Essentiality Prediction.*  
> GitHub Repository: https://github.com/YourUsername/GEP-GNN

---

## Acknowledgements

This work was inspired by:
- Kipf & Welling (2016) — Semi-Supervised Classification with Graph Convolutional Networks
- Velickovic et al. (2018) — Graph Attention Networks
- Xu et al. (2019) — How Powerful Are Graph Neural Networks? (GIN)
- Ying et al. (2018) — Hierarchical Graph Representation Learning with Differentiable Pooling (DiffPool)

---

## License

This project is released under the MIT License — feel free to use and modify with attribution.

---

⭐ If you find this repository useful, please consider starring it on GitHub!

