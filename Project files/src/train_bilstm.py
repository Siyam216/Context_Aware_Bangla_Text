import os
import sys
import time
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Ensure UTF-8 output encoding for Windows PowerShell
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Device configuration (CPU / CUDA)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned_data")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
PLOTS_DIR = os.path.join(BASE_DIR, "eda_plots")
RESULTS_PATH = os.path.join(BASE_DIR, "results_bilstm.json")

MAX_SEQ_LEN = 50
TOKEN_PATTERN = re.compile(r'[\u0980-\u09FFa-zA-Z0-9]+|[!?]')

# Task Configurations
TASKS = {
    "sentiment": {
        "dir": os.path.join(CLEANED_DIR, "sentiment"),
        "num_classes": 3,
        "class_names": ["Negative", "Neutral", "Positive"],
        "labels": [0, 1, 2],
        "checkpoint": "bilstm_sentiment.pt",
        "epochs": 3,
        "batch_size": 256,
        "lr": 0.001
    },
    "sarcasm": {
        "dir": os.path.join(CLEANED_DIR, "sarcasm"),
        "num_classes": 2,
        "class_names": ["Non-Sarcastic", "Sarcastic"],
        "labels": [0, 1],
        "checkpoint": "bilstm_sarcasm.pt",
        "epochs": 5,
        "batch_size": 128,
        "lr": 0.001
    },
    "hate_speech": {
        "dir": os.path.join(CLEANED_DIR, "hate_speech"),
        "num_classes": 2,
        "class_names": ["Non-Hate", "Hate Speech"],
        "labels": [0, 1],
        "checkpoint": "bilstm_hate.pt",
        "epochs": 4,
        "batch_size": 128,
        "lr": 0.001
    }
}


class StackedBiLSTMClassifier(nn.Module):
    """
    Stacked Bidirectional LSTM Classifier as taught in Lab 4.
    - Embedding: Initialized with pre-trained Word2Vec weights, fine-tunable (freeze=False)
    - BiLSTM: 2 stacked bidirectional LSTM layers with dropout
    - Linear Head: Projects concatenated forward and backward hidden states to class logits
    """
    def __init__(self, embed_matrix: torch.Tensor, hidden_dim: int = 64, num_layers: int = 2, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        vocab_size, embed_dim = embed_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            embed_matrix,
            freeze=False,
            padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        out, (h_n, c_n) = self.lstm(embedded)
        # Concatenate final forward and backward hidden states from the last LSTM layer
        # h_n shape: (num_layers * 2, batch_size, hidden_dim)
        forward_h = h_n[-2, :, :]   # (batch_size, hidden_dim)
        backward_h = h_n[-1, :, :]  # (batch_size, hidden_dim)
        bi_hidden = torch.cat((forward_h, backward_h), dim=1)  # (batch_size, hidden_dim * 2)
        bi_hidden = self.dropout(bi_hidden)
        logits = self.fc(bi_hidden)  # (batch_size, num_classes)
        return logits


def encode_texts(texts: list, word2idx: dict, max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
    encoded_list = []
    unk_id = word2idx.get("<UNK>", 1)
    pad_id = word2idx.get("<PAD>", 0)

    for text in texts:
        tokens = TOKEN_PATTERN.findall(str(text))
        indices = [word2idx.get(t, unk_id) for t in tokens[:max_len]]
        if len(indices) < max_len:
            indices += [pad_id] * (max_len - len(indices))
        encoded_list.append(indices)

    return torch.tensor(encoded_list, dtype=torch.long)


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    # Inverse class frequency weighting
    weights = total / (num_classes * np.maximum(counts, 1).astype(float))
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        # Gradient clipping for LSTM stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, (correct / total) * 100


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    _, _, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    avg_loss = total_loss / len(all_labels)

    return avg_loss, acc, f1_macro, all_preds, all_labels, all_probs


def train_task_bilstm(task_name: str, config: dict, embed_matrix: torch.Tensor, word2idx: dict):
    print(f"\n=======================================================")
    print(f"[*] Training Stacked BiLSTM: {task_name.upper()}")
    print(f"=======================================================")
    
    # 1. Load Datasets
    task_dir = config["dir"]
    train_df = pd.read_csv(os.path.join(task_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(task_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(task_dir, "test.csv"))
    
    print(f"  [+] Loaded splits: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    # 2. Encode Texts
    print(f"  [+] Encoding sentences into token ID sequences (max_len={MAX_SEQ_LEN})...")
    X_train = encode_texts(train_df['text'].tolist(), word2idx)
    y_train = torch.tensor(train_df['label'].values, dtype=torch.long)

    X_val = encode_texts(val_df['text'].tolist(), word2idx)
    y_val = torch.tensor(val_df['label'].values, dtype=torch.long)

    X_test = encode_texts(test_df['text'].tolist(), word2idx)
    y_test = torch.tensor(test_df['label'].values, dtype=torch.long)

    # 3. DataLoaders
    batch_size = config["batch_size"]
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # 4. Model, Loss, Optimizer
    class_weights = compute_class_weights(y_train.numpy(), config["num_classes"]).to(device)
    print(f"  [+] Balanced Class Weights: {class_weights.cpu().numpy().round(3).tolist()}")
    
    model = StackedBiLSTMClassifier(
        embed_matrix=embed_matrix,
        hidden_dim=64,
        num_layers=2,
        num_classes=config["num_classes"],
        dropout=0.3
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # 5. Training Loop
    checkpoint_path = os.path.join(MODELS_DIR, config["checkpoint"])
    best_val_f1 = -1.0
    epochs = config["epochs"]
    t_train_start = time.time()

    print(f"  [+] Training for {epochs} epochs on {device}...")
    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _, _ = evaluate(model, val_loader, criterion, device)
        ep_time = time.time() - ep_start

        print(f"      Epoch {epoch}/{epochs} ({ep_time:.1f}s) | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% Macro-F1: {val_f1*100:.2f}%")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": config["num_classes"],
                "hidden_dim": 64,
                "num_layers": 2,
                "embed_dim": embed_matrix.shape[1],
                "vocab_size": embed_matrix.shape[0],
                "class_names": config["class_names"],
                "val_macro_f1": val_f1
            }, checkpoint_path)

    train_total_time = time.time() - t_train_start
    print(f"  [+] Training completed in {train_total_time:.1f}s. Best Val Macro-F1: {best_val_f1*100:.2f}%")

    # 6. Load Best Checkpoint & Evaluate on Test Set
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loss, test_acc, test_f1_macro, test_preds, test_labels, _ = evaluate(model, test_loader, criterion, device)
    
    test_prec_macro, test_rec_macro, _, _ = precision_recall_fscore_support(
        test_labels, test_preds, average='macro', zero_division=0
    )
    test_prec_wt, test_rec_wt, test_f1_wt, _ = precision_recall_fscore_support(
        test_labels, test_preds, average='weighted', zero_division=0
    )

    # Class-wise metrics
    p_per_class, r_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        test_labels, test_preds, labels=config["labels"], average=None, zero_division=0
    )

    class_metrics = {}
    for idx, cname in enumerate(config["class_names"]):
        class_metrics[cname] = {
            "precision": float(round(p_per_class[idx], 4)),
            "recall": float(round(r_per_class[idx], 4)),
            "f1_score": float(round(f1_per_class[idx], 4)),
            "support": int(support_per_class[idx])
        }

    cm = confusion_matrix(test_labels, test_preds, labels=config["labels"])

    print(f"\n  [+] TEST RESULTS for {task_name.upper()} (BiLSTM):")
    print(f"      - Test Accuracy:     {test_acc*100:.2f}%")
    print(f"      - Test Macro F1:     {test_f1_macro*100:.2f}%")
    print(f"      - Test Weighted F1:  {test_f1_wt*100:.2f}%")
    print(f"      - Test Macro Recall: {test_rec_macro*100:.2f}%")
    print("\n  [+] Detailed Classification Report (Test Set):")
    print(classification_report(test_labels, test_preds, target_names=config["class_names"], digits=4))
    print(f"  [+] Saved best model weights -> {checkpoint_path} ({os.path.getsize(checkpoint_path)/(1024*1024):.2f} MB)")

    return {
        "task": task_name,
        "training_time_sec": round(train_total_time, 2),
        "validation_metrics": {
            "best_macro_f1": float(round(best_val_f1, 4))
        },
        "test_metrics": {
            "accuracy": float(round(test_acc, 4)),
            "macro_precision": float(round(test_prec_macro, 4)),
            "macro_recall": float(round(test_rec_macro, 4)),
            "macro_f1": float(round(test_f1_macro, 4)),
            "weighted_f1": float(round(test_f1_wt, 4)),
            "class_wise": class_metrics
        },
        "confusion_matrix": cm.tolist()
    }, cm, model


def plot_all_confusion_matrices(results_dict, cm_dict):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plt.subplots_adjust(wspace=0.35)

    for idx, (task_name, config) in enumerate(TASKS.items()):
        ax = axes[idx]
        cm = np.array(cm_dict[task_name])
        class_names = config["class_names"]

        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1f}%)"

        cmap = "Blues" if task_name == "sentiment" else ("Purples" if task_name == "sarcasm" else "Reds")
        sns.heatmap(cm, annot=annot, fmt='', cmap=cmap, cbar=False,
                    xticklabels=class_names, yticklabels=class_names, ax=ax,
                    annot_kws={"size": 11, "weight": "bold"})

        test_f1 = results_dict[task_name]["test_metrics"]["macro_f1"] * 100
        test_acc = results_dict[task_name]["test_metrics"]["accuracy"] * 100

        title_str = f"BiLSTM {task_name.replace('_', ' ').title()} Matrix\nAcc: {test_acc:.1f}% | Macro-F1: {test_f1:.1f}%"
        ax.set_title(title_str, fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("Predicted Label", fontsize=11, fontweight='bold')
        ax.set_ylabel("True Label", fontsize=11, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    output_fig_path = os.path.join(PLOTS_DIR, "confusion_matrices_bilstm.png")
    plt.savefig(output_fig_path, dpi=300)
    plt.close()
    print(f"\n[+] BiLSTM Confusion Matrix Plot saved -> {output_fig_path}")


def main():
    print("=" * 65)
    print("  PHASE 4.2: PYTORCH STACKED BILSTM TRAINING PIPELINE")
    print("=" * 65)
    t_all_start = time.time()

    # Load Word2Vec and Vocab
    embed_path = os.path.join(MODELS_DIR, "bangla_word2vec.pt")
    vocab_path = os.path.join(MODELS_DIR, "word2idx.json")

    if not os.path.exists(embed_path) or not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Missing Word2Vec artifacts in {MODELS_DIR}. Run train_word2vec.py first!")

    print(f"[+] Loading Pretrained Word2Vec Embeddings from {embed_path}...")
    embed_matrix = torch.load(embed_path, map_location=device)
    with open(vocab_path, "r", encoding="utf-8") as f:
        word2idx = json.load(f)

    print(f"  [+] Embedding Shape: {embed_matrix.shape} (Vocab={len(word2idx):,}, Dim={embed_matrix.shape[1]})")

    all_results = {}
    all_cms = {}

    for task_name, config in TASKS.items():
        res, cm, _ = train_task_bilstm(task_name, config, embed_matrix, word2idx)
        all_results[task_name] = res
        all_cms[task_name] = cm

    # Save results JSON
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Master BiLSTM Results saved -> {RESULTS_PATH}")

    # Plot Confusion Matrices
    plot_all_confusion_matrices(all_results, all_cms)

    total_time = time.time() - t_all_start
    print(f"\n=======================================================")
    print(f"[✔] PHASE 4 BILSTM PIPELINE COMPLETED in {total_time:.1f}s")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
