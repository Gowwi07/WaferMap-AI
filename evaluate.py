"""
evaluate.py
===========
Standalone test-set evaluation script.

Run this after training to generate final metrics, confusion matrix,
classification report, and F1 bar charts from the saved best model.

Usage:
    python evaluate.py
"""

import os, sys, json, time
import torch
import numpy as np
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from src.dataset import CLASS_NAMES, NUM_CLASSES, get_dataloaders
from src.model import build_model

MODEL_PATH  = os.path.join(ROOT_DIR, "models", "efficientnet_wafer.pth")
REPORT_DIR  = os.path.join(ROOT_DIR, "reports")
METRICS_PATH = os.path.join(REPORT_DIR, "phase2_metrics.json")
os.makedirs(REPORT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[Eval] Device: {DEVICE}")

# ── Load model ───────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model not found at {MODEL_PATH}")
    print("        Please finish training first: python src/train.py")
    sys.exit(1)

model = build_model(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"[Eval] Model loaded from {MODEL_PATH}")

# ── Load test data ────────────────────────────────────────────────────────────
print("[Eval] Loading test split...")
_, _, test_loader, _ = get_dataloaders(batch_size=64, num_workers=0)
print(f"[Eval] Test batches: {len(test_loader)}")

# ── Run inference ─────────────────────────────────────────────────────────────
all_preds, all_labels = [], []
total, correct = 0, 0

with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="[Test]"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total

# ── Print report ──────────────────────────────────────────────────────────────
report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
macro_f1 = report["macro avg"]["f1-score"]

print("\n" + "="*60)
print("  TEST SET RESULTS")
print("="*60)
print(f"  Overall Accuracy : {test_acc*100:.2f}%")
print(f"  Macro F1 Score   : {macro_f1:.4f}")
print()
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# ── Save JSON metrics ─────────────────────────────────────────────────────────
metrics = {
    "test_accuracy": round(test_acc * 100, 2),
    "macro_f1": round(macro_f1, 4),
    "device": str(DEVICE),
    "per_class": {
        cls: {
            "precision": round(report[cls]["precision"], 4),
            "recall":    round(report[cls]["recall"],    4),
            "f1":        round(report[cls]["f1-score"],  4),
            "support":   int(report[cls]["support"]),
        }
        for cls in CLASS_NAMES if cls in report
    }
}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"[Save] Metrics JSON --> {METRICS_PATH}")

# ── Confusion Matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(11, 9))
fig.patch.set_facecolor("#0a0a0a")
ax.set_facecolor("#0a0a0a")
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, linecolor="#222",
            cbar_kws={"shrink": 0.8}, ax=ax,
            annot_kws={"size": 8, "color": "white"})
ax.set_xlabel("Predicted", color="white", fontsize=11, labelpad=8)
ax.set_ylabel("True Label", color="white", fontsize=11, labelpad=8)
ax.set_title("Normalized Confusion Matrix — WaferMap AI", color="white", fontsize=13, pad=12)
ax.tick_params(colors="white", labelsize=9)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
cm_path = os.path.join(REPORT_DIR, "phase2_confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"[Save] Confusion matrix --> {cm_path}")

# ── F1 Bar Chart ──────────────────────────────────────────────────────────────
f1_scores = [report[c]["f1-score"] for c in CLASS_NAMES if c in report]
colors = ["#e74c3c" if f < 0.7 else "#f5b041" if f < 0.85 else "#2ecc71" for f in f1_scores]

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#0a0a0a")
ax.set_facecolor("#0a0a0a")
bars = ax.barh([c for c in CLASS_NAMES if c in report], f1_scores, color=colors, height=0.6)
ax.set_xlim(0, 1.05)
ax.set_xlabel("F1 Score", color="white", fontsize=11)
ax.set_title("Per-Class F1 Scores — WaferMap AI", color="white", fontsize=13, pad=10)
ax.tick_params(colors="white")
ax.spines[:].set_color("#333")
for bar, val in zip(bars, f1_scores):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", color="white", fontsize=9)
ax.axvline(x=macro_f1, color="#7c5cbf", linestyle="--", linewidth=1.5,
           label=f"Macro avg = {macro_f1:.3f}")
ax.legend(facecolor="#111", labelcolor="white")
plt.tight_layout()
f1_path = os.path.join(REPORT_DIR, "phase2_f1_scores.png")
plt.savefig(f1_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"[Save] F1 bar chart   --> {f1_path}")

print("\n" + "="*60)
print("  Evaluation complete!")
print(f"  Test Accuracy : {test_acc*100:.2f}%")
print(f"  Macro F1      : {macro_f1:.4f}")
print("="*60)
