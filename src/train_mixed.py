"""
src/train_mixed.py
==================
Phase 2B — Multi-Label Defect Detection Training
Trains an EfficientNet-B0 on MixedWM38 for multi-label classification.

Key differences from single-label training:
  - Loss: BCEWithLogitsLoss  (not CrossEntropyLoss)
  - Output: Sigmoid threshold (not Softmax + argmax)
  - Metric: Macro F1 + Hamming Loss (not simple accuracy)
  - Labels: float vector (not integer class index)
"""

import os, sys, time, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, hamming_loss, classification_report
from tqdm import tqdm
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.dataset_mixed import get_mixed_dataloaders, MIXED_CLASS_NAMES, NUM_MIXED_CLASSES
from src.model import build_model

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(ROOT_DIR, "models", "mixed_model.pth")
REPORT_DIR  = os.path.join(ROOT_DIR, "reports")
METRICS_PATH = os.path.join(REPORT_DIR, "mixed_metrics.json")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPOCHS      = 10     # Quick Config
BATCH_SIZE  = 32     # Matches single-label batch size for consistency
LR_BACKBONE = 1e-4   # Conservative LR for smaller dataset stability
LR_HEAD     = 1e-4   # Consistent head LR for fine-tuning
DROPOUT     = 0.4    # Slightly more dropout for multi-label complexity
NUM_WORKERS = 0      # Windows safe
THRESHOLD   = 0.5    # Sigmoid threshold for positive prediction
EARLY_STOP  = 10     # Disable early stopping for full run

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Train-Multi] Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[Train-Multi] GPU: {torch.cuda.get_device_name(0)}")


# ── Training function ──────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)                      # raw logits (B, 8)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

        # For metric: apply sigmoid + threshold
        preds = (torch.sigmoid(outputs).detach().cpu().numpy() >= THRESHOLD).astype(int)
        all_preds.append(preds)
        all_targets.append(labels.cpu().numpy().astype(int))
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    all_preds   = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


# ── Evaluation function ────────────────────────────────────────────────────────
def eval_epoch(model, loader, criterion, split="Val"):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"[{split}]", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)

            preds = (torch.sigmoid(outputs).cpu().numpy() >= THRESHOLD).astype(int)
            all_preds.append(preds)
            all_targets.append(labels.cpu().numpy().astype(int))

    all_preds   = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    h_loss   = hamming_loss(all_targets, all_preds)
    return avg_loss, macro_f1, h_loss, all_preds, all_targets


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    print("\n" + "="*60)
    print("  WaferMapAI — Phase 2B: Multi-Label Training (MixedWM38)")
    print("="*60)

    # ── 1. Data ────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, pos_weight = get_mixed_dataloaders(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    # ── 2. Model ───────────────────────────────────────────────────────────────
    # Reuse same EfficientNet-B0 backbone, BUT with 8 output classes (not 9)
    model = build_model(num_classes=NUM_MIXED_CLASSES, dropout=DROPOUT, pretrained=True)
    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] EfficientNet-B0 | Output classes: {NUM_MIXED_CLASSES} | Params: {total_params:,}")

    # ── 3. Loss: BCEWithLogitsLoss (multi-label standard) ─────────────────────
    pos_weight = pos_weight.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── 4. Optimizer ───────────────────────────────────────────────────────────
    backbone_params   = [p for n, p in model.named_parameters() if "classifier" not in n]
    classifier_params = [p for n, p in model.named_parameters() if "classifier"     in n]
    optimizer = optim.AdamW([
        {"params": backbone_params,   "lr": LR_BACKBONE},
        {"params": classifier_params, "lr": LR_HEAD},
    ], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── 5. Training loop ───────────────────────────────────────────────────────
    history = {"train_loss": [], "train_f1": [], "val_loss": [], "val_f1": [], "val_hamming": []}
    best_val_f1 = 0.0
    no_improve  = 0
    
    # Update max epochs for a more thorough run
    MAX_EPOCHS = 10 
    print(f"\n[Train] Starting {MAX_EPOCHS}-epoch run (frozen backbone for 3 epochs)\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        ep_start = time.time()
        
        # Solution 4: Stage 1 Freezing (Epochs 1-3)
        if hasattr(model, "features"):
            if epoch <= 3:
                for param in model.features.parameters():
                    param.requires_grad = False
                print(f"  [Stage 1] Backbone Frozen (Epoch {epoch})")
            else:
                for param in model.features.parameters():
                    param.requires_grad = True
                if epoch == 4:
                    print(f"  [Stage 2] Backbone Unfrozen (Epoch {epoch})")

        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, epoch, MAX_EPOCHS)
        val_loss, val_f1, val_hamming, _, _ = eval_epoch(model, val_loader, criterion, "Val")
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_hamming"].append(val_hamming)

        ep_time = time.time() - ep_start
        print(
            f"  Epoch {epoch:>2}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f}  F1: {val_f1:.4f}  Hamming: {val_hamming:.4f} | "
            f"Time: {ep_time:.0f}s"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  [**] Best model saved (val_f1={val_f1:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            print(f"  [--] No improvement for {no_improve}/{EARLY_STOP} epochs (best={best_val_f1:.4f})")
            if no_improve >= EARLY_STOP:
                print(f"\n  [Early Stop] Stopping at epoch {epoch}.")
                break

    total_time = time.time() - start_time
    print(f"\n[Train] Complete in {total_time/60:.1f} min | Best Val F1: {best_val_f1:.4f}")

    # ── 6. Test evaluation ─────────────────────────────────────────────────────
    print("\n[Eval] Loading best model for test evaluation...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    test_loss, test_f1, test_hamming, test_preds, test_targets = eval_epoch(
        model, test_loader, criterion, "Test"
    )

    report = classification_report(
        test_targets, test_preds,
        target_names=MIXED_CLASS_NAMES, output_dict=True, zero_division=0
    )
    print("\n" + "="*60)
    print("  TEST SET RESULTS (Multi-Label)")
    print("="*60)
    print(f"  Macro F1     : {test_f1:.4f}")
    print(f"  Hamming Loss : {test_hamming:.4f}  (lower=better, 0=perfect)")
    print()
    print(classification_report(test_targets, test_preds, target_names=MIXED_CLASS_NAMES, zero_division=0))

    # ── 7. Save metrics ────────────────────────────────────────────────────────
    metrics = {
        "macro_f1":      round(test_f1, 4),
        "hamming_loss":  round(test_hamming, 4),
        "best_val_f1":   round(best_val_f1, 4),
        "epochs_run":    epoch,
        "device":        str(DEVICE),
        "training_minutes": round(total_time/60, 1),
        "threshold":     THRESHOLD,
        "history":       history,
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"],    4),
                "f1":        round(report[cls]["f1-score"],  4),
                "support":   int(report[cls]["support"]),
            }
            for cls in MIXED_CLASS_NAMES if cls in report
        }
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Save] Metrics --> {METRICS_PATH}")
    print(f"[Save] Model   --> {MODEL_PATH}")

    print("\n" + "="*60)
    print("  Phase 2B COMPLETE!")
    print(f"  To continue training, increase EPOCHS and run again.")
    print("="*60)


if __name__ == "__main__":
    main()
