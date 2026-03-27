"""
src/train.py
============
Phase 2 - Training Pipeline

What this script does (plain English):
1. Loads the wafer map dataset (via dataset.py)
2. Builds the EfficientNet-B0 model (via model.py)
3. Trains for N epochs — each epoch goes through all training data once
4. After each epoch, checks performance on the validation set
5. Saves the BEST model (the one with highest val accuracy)
6. At the end, evaluates on the test set and saves metrics

Key terms explained in comments throughout.

Run with:
    python src/train.py
"""

import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to path so imports work
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.dataset import get_dataloaders, CLASS_NAMES, NUM_CLASSES
from src.model   import build_model, model_summary

# ── Config ────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(ROOT_DIR, "models")
REPORT_DIR = os.path.join(ROOT_DIR, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

MODEL_PATH  = os.path.join(MODELS_DIR, "efficientnet_wafer.pth")
METRICS_PATH = os.path.join(REPORT_DIR, "phase2_metrics.json")

# ── Hyperparameters ───────────────────────────────────────────────────────────
# GPU-optimized settings matching PPT specification for >96% accuracy
EPOCHS       = 30    # Full 30 epochs on GPU (2 hrs on T4, faster on RTX 3050)
BATCH_SIZE   = 32    # Larger batch for GPU throughput
LR_BACKBONE  = 1e-4  # Adam LR matching PPT spec
LR_HEAD      = 1e-3  # Larger LR for classifier head
DROPOUT      = 0.3   # 30% dropout for regularization
NUM_WORKERS  = 0     # MUST BE 0 on Windows to prevent PyTorch DataLoader deadlocks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Train] Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Train] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


# ── Training functions ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, epoch, total_epochs):
    """
    Train the model for one epoch.

    An EPOCH = one complete pass through the entire training dataset.
    If we have 50,000 training images and batch_size=32:
      50,000 / 32 = ~1,563 batches per epoch

    For each batch:
      1. Forward pass: model sees the images and makes predictions
      2. Loss: measure how wrong the predictions are
      3. Backward pass: compute how to adjust weights to improve
      4. Optimizer step: actually update the weights
    """
    model.train()  # Training mode: dropout is active, batch norm uses batch stats
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]",
                ncols=80, leave=False)

    for imgs, labels in pbar:
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()        # Clear previous gradients

        outputs = model(imgs)        # Forward pass: [batch, 9] logits
        loss    = criterion(outputs, labels)  # How wrong are we?

        loss.backward()              # Backprop: compute gradients
        optimizer.step()             # Update weights

        # Track metrics
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.3f}"})

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, phase="Val"):
    """
    Evaluate the model on val or test set.

    @torch.no_grad() = do NOT compute gradients (saves memory, speeds up evaluation)
    model.eval()     = evaluation mode: dropout is OFF, consistent predictions
    """
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for imgs, labels in tqdm(loader, desc=f"[{phase}]", ncols=80, leave=False):
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(imgs)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, all_preds, all_labels


# ── Plotting functions ────────────────────────────────────────────────────────

def plot_training_curves(history, save_path):
    """Plot loss and accuracy over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")

    epochs_range = range(1, len(history["train_loss"]) + 1)

    for ax in [ax1, ax2]:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#0f3460")

    # Loss curve
    ax1.plot(epochs_range, history["train_loss"], "o-", color="#e74c3c",
             label="Train Loss", linewidth=2)
    ax1.plot(epochs_range, history["val_loss"],   "o-", color="#00d2ff",
             label="Val Loss",   linewidth=2, linestyle="--")
    ax1.set_title("Loss vs Epoch", color="white", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", color="#aaa")
    ax1.set_ylabel("Loss",  color="#aaa")
    ax1.legend(facecolor="#0f3460", labelcolor="white")
    ax1.yaxis.label.set_color("#aaa")
    ax1.xaxis.label.set_color("#aaa")

    # Accuracy curve
    ax2.plot(epochs_range, [a*100 for a in history["train_acc"]], "o-",
             color="#e74c3c", label="Train Acc", linewidth=2)
    ax2.plot(epochs_range, [a*100 for a in history["val_acc"]], "o-",
             color="#00d2ff", label="Val Acc", linewidth=2, linestyle="--")
    ax2.set_title("Accuracy vs Epoch", color="white", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", color="#aaa")
    ax2.set_ylabel("Accuracy (%)", color="#aaa")
    ax2.legend(facecolor="#0f3460", labelcolor="white")
    ax2.yaxis.label.set_color("#aaa")
    ax2.xaxis.label.set_color("#aaa")

    plt.suptitle("EfficientNet-B0 Training Curves", color="white",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Save] Training curves --> {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    # Normalize by row (true label) so each cell shows % within that class
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        cmap="Blues", ax=ax,
        linewidths=0.5, linecolor="#0f3460",
    )
    ax.set_title("Confusion Matrix (Normalized)", color="white",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", color="#aaa")
    ax.set_ylabel("True Label", color="#aaa")
    ax.tick_params(colors="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Save] Confusion matrix --> {save_path}")


def plot_f1_bars(report_dict, save_path):
    """Plot per-class F1 scores as horizontal bars."""
    classes = [c for c in CLASS_NAMES if c in report_dict]
    f1s     = [report_dict[c]["f1-score"] for c in classes]

    colors = [
        "#e74c3c" if f < 0.7 else "#f39c12" if f < 0.85 else "#27ae60"
        for f in f1s
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    bars = ax.barh(classes, f1s, color=colors, edgecolor="#0f3460")
    ax.set_xlim(0, 1.05)

    for bar, val in zip(bars, f1s):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", color="white", fontsize=10)

    ax.set_title("Per-Class F1 Score", color="white", fontsize=13, fontweight="bold")
    ax.set_xlabel("F1 Score", color="#aaa")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#0f3460")
    ax.xaxis.label.set_color("#aaa")

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color="#e74c3c", label="F1 < 0.70  (Needs work)"),
        Patch(color="#f39c12", label="F1 0.70-0.85 (Good)"),
        Patch(color="#27ae60", label="F1 > 0.85  (Excellent)"),
    ]
    ax.legend(handles=legend, facecolor="#0f3460", labelcolor="white",
              fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Save] F1 bar chart    --> {save_path}")


# ── Main training loop ────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 60)
    print("  WaferMapAI - Phase 2: Model Training")
    print("=" * 60)
    start_time = time.time()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    # ── 2. Build model ────────────────────────────────────────────────────────
    model = build_model(num_classes=NUM_CLASSES, dropout=DROPOUT, pretrained=True)
    model_summary(model)
    model = model.to(DEVICE)

    # ── 3. Loss function ──────────────────────────────────────────────────────
    # CrossEntropyLoss: measures how wrong our predictions are.
    # weight= gives rare classes higher penalty (class imbalance fix)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    # ── 4. Optimizer ──────────────────────────────────────────────────────────
    # We use different learning rates:
    # - Backbone (pretrained EfficientNet layers): small LR (already well-trained)
    # - Classifier head (our new layers): larger LR (learning from scratch)
    backbone_params   = [p for n, p in model.named_parameters()
                         if "classifier" not in n]
    classifier_params = [p for n, p in model.named_parameters()
                         if "classifier" in n]

    optimizer = optim.AdamW([
        {"params": backbone_params,   "lr": LR_BACKBONE},
        {"params": classifier_params, "lr": LR_HEAD},
    ], weight_decay=1e-4)

    # Scheduler: slowly reduce LR over epochs (cosine schedule)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── 5. Training loop ──────────────────────────────────────────────────────
    history    = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        ep_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, EPOCHS
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, "Val")
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        ep_time = time.time() - ep_start
        print(
            f"  Epoch {epoch:>2}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.1f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:.1f}% | "
            f"Time: {ep_time:.0f}s"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  [**] New best model saved (val_acc={val_acc*100:.1f}%)")

    total_time = time.time() - start_time
    print(f"\n[Train] Training complete in {total_time/60:.1f} minutes.")
    print(f"[Train] Best val accuracy: {best_val_acc*100:.1f}%")

    # ── 6. Final evaluation on test set ─────────────────────────────────────
    print("\n[Eval] Loading best model for test evaluation...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    _, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, "Test"
    )

    # Detailed classification report
    report = classification_report(
        test_labels, test_preds,
        target_names=CLASS_NAMES, output_dict=True
    )
    macro_f1 = report["macro avg"]["f1-score"]

    print()
    print("=" * 60)
    print("  TEST SET RESULTS")
    print("=" * 60)
    print(f"  Overall Accuracy : {test_acc*100:.2f}%")
    print(f"  Macro F1 Score   : {macro_f1:.4f}")
    print()
    print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))

    # ── 7. Save metrics to JSON ───────────────────────────────────────────────
    metrics = {
        "test_accuracy": round(test_acc * 100, 2),
        "macro_f1":      round(macro_f1, 4),
        "training_epochs": EPOCHS,
        "best_val_accuracy": round(best_val_acc * 100, 2),
        "device": str(DEVICE),
        "training_minutes": round(total_time / 60, 1),
        "history": history,
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"],    4),
                "f1":        round(report[cls]["f1-score"],  4),
                "support":   report[cls]["support"],
            }
            for cls in CLASS_NAMES if cls in report
        }
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Save] Metrics JSON --> {METRICS_PATH}")

    # ── 8. Generate plots ─────────────────────────────────────────────────────
    plot_training_curves(history,
        os.path.join(REPORT_DIR, "phase2_training_curves.png"))
    plot_confusion_matrix(test_labels, test_preds,
        os.path.join(REPORT_DIR, "phase2_confusion_matrix.png"))
    plot_f1_bars(report,
        os.path.join(REPORT_DIR, "phase2_f1_scores.png"))

    print()
    print("=" * 60)
    print(f"  Phase 2 COMPLETE!")
    print(f"  Model saved to   : {MODEL_PATH}")
    print(f"  Reports saved to : {REPORT_DIR}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
