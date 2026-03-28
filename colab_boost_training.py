"""
colab_boost_training.py
=======================
FAST BOOST TRAINING — Run this in Google Colab to retrain both models with
higher accuracy. Estimated time: ~25 min (single) + ~5 min (mixed) on T4 GPU.

Key improvements over previous training:
- Single-label: 25 epochs (was 10), 1000 samples/class (was 500),
  cosine warmup schedule, lower LR = more stable convergence
- Mixed: 20 epochs (was 10), full dataset sampling
- Both: data stays at 224x224 NEAREST (matches inference exactly)
"""

# ── [0] COLAB SETUP ──────────────────────────────────────────────────────────
import subprocess, os, sys

def run(cmd): subprocess.run(cmd, shell=True, check=True)

# Install dependencies
run("pip install -q efficientnet-pytorch torchvision seaborn tqdm scikit-learn")

# Clone repo
if not os.path.exists("WaferMap-AI"):
    run("git clone https://github.com/Gowwi07/WaferMap-AI.git")

os.chdir("WaferMap-AI")
sys.path.insert(0, ".")

# Mount Drive for data
from google.colab import drive
drive.mount("/content/drive")

# Copy data files from Drive (adjust path if needed)
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

DRIVE_DATA = "/content/drive/MyDrive/wafer_data"  # ← CHANGE THIS TO YOUR PATH
if os.path.exists(DRIVE_DATA):
    run(f"cp {DRIVE_DATA}/LSWMD.pkl {DATA_DIR}/LSWMD.pkl")
    run(f"cp {DRIVE_DATA}/MixedWM38.npz {DATA_DIR}/MixedWM38.npz")
else:
    print("[WARN] Data not found at", DRIVE_DATA)
    print("       Upload LSWMD.pkl and MixedWM38.npz to your Drive and update DRIVE_DATA path.")

# ── [1] IMPORTS ───────────────────────────────────────────────────────────────
import json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report
from PIL import Image

from src.dataset import CLASS_NAMES, NUM_CLASSES, load_raw_dataframe, split_dataframe, WaferMapDataset, get_class_weights, IMG_SIZE, wafer_to_image
from src.dataset_mixed import MIXED_CLASS_NAMES, NUM_MIXED_CLASSES, wafer_array_to_pil, MIXED_TRANSFORM, MixedWM38Dataset
from src.model import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ── [2] EVAL TRANSFORM — must match training ─────────────────────────────────
EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.3),
    transforms.RandomApply([transforms.RandomRotation((180, 180))], p=0.3),
    transforms.RandomApply([transforms.RandomRotation((270, 270))], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ═══════════════════════════════════════════════════════════════════════════════
# PART A: SINGLE-LABEL MODEL  (wafer_model.pth)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PART A: Single-Label Training  (25 epochs)")
print("=" * 60)

EPOCHS_SINGLE   = 25
BATCH_SIZE      = 32
SAMPLES_PER_CLS = 1000   # Double from previous 500

# Load & balance data
df = load_raw_dataframe()

# Oversample to SAMPLES_PER_CLS min per class
from collections import Counter
import pandas as pd

parts = []
for cls in CLASS_NAMES:
    cls_df = df[df["label"] == cls]
    n = min(len(cls_df), SAMPLES_PER_CLS)
    if n > 0:
        parts.append(cls_df.sample(n=n, random_state=42))
df_bal = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Balanced dataset: {len(df_bal)} samples")
print(df_bal["label"].value_counts().to_string())

train_df, val_df, test_df = split_dataframe(df_bal)

train_ds = WaferMapDataset(train_df, transform=TRAIN_TRANSFORM)
val_ds   = WaferMapDataset(val_df,   transform=EVAL_TRANSFORM)
test_ds  = WaferMapDataset(test_df,  transform=EVAL_TRANSFORM)

cw = get_class_weights(train_df)
sample_weights = [cw[{n: i for i, n in enumerate(CLASS_NAMES)}[lbl]] for lbl in train_df["label"]]
sampler = WeightedRandomSampler(sample_weights, len(train_df), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,  num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,  num_workers=2)

# Model
model = build_model(num_classes=NUM_CLASSES, dropout=0.3, pretrained=True).to(DEVICE)
class_weights_tensor = cw.to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)

optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
scheduler = OneCycleLR(optimizer, max_lr=3e-4, steps_per_epoch=len(train_loader), epochs=EPOCHS_SINGLE)

MODEL_PATH_SINGLE = "models/wafer_model.pth"
best_val_acc = 0.0
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

for epoch in range(1, EPOCHS_SINGLE + 1):
    # Progressive unfreezing: freeze backbone first 5 epochs
    if epoch <= 5:
        for p in model.features.parameters(): p.requires_grad = False
    else:
        for p in model.features.parameters(): p.requires_grad = True

    model.train()
    t_loss, t_preds, t_lbls = 0, [], []
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        scheduler.step()
        t_loss += loss.item() * imgs.size(0)
        t_preds.extend(out.argmax(1).cpu().numpy())
        t_lbls.extend(lbls.cpu().numpy())

    t_acc = accuracy_score(t_lbls, t_preds)

    model.eval()
    v_loss, v_preds, v_lbls = 0, [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            out = model(imgs)
            v_loss += criterion(out, lbls).item() * imgs.size(0)
            v_preds.extend(out.argmax(1).cpu().numpy())
            v_lbls.extend(lbls.cpu().numpy())
    v_acc = accuracy_score(v_lbls, v_preds)
    avg_t = t_loss / len(train_ds); avg_v = v_loss / len(val_ds)
    history["train_loss"].append(avg_t); history["train_acc"].append(t_acc)
    history["val_loss"].append(avg_v);   history["val_acc"].append(v_acc)
    print(f"  E{epoch:02d}/{EPOCHS_SINGLE} | TLoss:{avg_t:.4f} TAcc:{t_acc*100:.1f}% | VLoss:{avg_v:.4f} VAcc:{v_acc*100:.1f}%")
    if v_acc > best_val_acc:
        best_val_acc = v_acc
        torch.save(model.state_dict(), MODEL_PATH_SINGLE)
        print(f"       ✅ Saved (best val={best_val_acc*100:.1f}%)")

# Test
model.load_state_dict(torch.load(MODEL_PATH_SINGLE, map_location=DEVICE))
model.eval(); te_preds, te_lbls = [], []
with torch.no_grad():
    for imgs, lbls in test_loader:
        out = model(imgs.to(DEVICE))
        te_preds.extend(out.argmax(1).cpu().numpy())
        te_lbls.extend(lbls.cpu().numpy())

te_acc = accuracy_score(te_lbls, te_preds)
report = classification_report(te_lbls, te_preds, target_names=CLASS_NAMES, output_dict=True)
macro_f1 = report["macro avg"]["f1-score"]
print(f"\n✅ SINGLE-LABEL TEST: Acc={te_acc*100:.2f}%  MacroF1={macro_f1:.4f}")
print(classification_report(te_lbls, te_preds, target_names=CLASS_NAMES))

metrics_single = {
    "test_accuracy": round(te_acc*100, 2),
    "macro_f1": round(macro_f1, 4),
    "training_epochs": EPOCHS_SINGLE,
    "best_val_accuracy": round(best_val_acc*100, 2),
    "device": str(DEVICE),
    "history": history,
    "per_class": {
        cls: {"precision": round(report[cls]["precision"],4),
              "recall": round(report[cls]["recall"],4),
              "f1": round(report[cls]["f1-score"],4),
              "support": report[cls]["support"]}
        for cls in CLASS_NAMES if cls in report
    }
}
with open("reports/phase2_metrics.json", "w") as f: json.dump(metrics_single, f, indent=2)

# ═══════════════════════════════════════════════════════════════════════════════
# PART B: MULTI-LABEL MODEL  (mixed_model.pth)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PART B: Multi-Label Training  (20 epochs)")
print("=" * 60)

EPOCHS_MIXED = 20
BATCH_MIXED  = 64

data = np.load("data/MixedWM38.npz")
arrays = data["arr_0"]; labels = data["arr_1"]; N = len(arrays)
print(f"Loaded {N:,} mixed samples")

# Full split (no subsampling — use all data)
n_test = int(N * 0.2); n_val = int(N * 0.1); n_train = N - n_test - n_val
from torch.utils.data import random_split
full_ds = MixedWM38Dataset(arrays, labels.astype(np.float32))
gen = torch.Generator().manual_seed(42)
tr_ds, va_ds, te_ds = random_split(full_ds, [n_train, n_val, n_test], generator=gen)
tr_loader = DataLoader(tr_ds, batch_size=BATCH_MIXED, shuffle=True,  num_workers=2, pin_memory=True)
va_loader = DataLoader(va_ds, batch_size=BATCH_MIXED, shuffle=False, num_workers=2)
te_loader = DataLoader(te_ds, batch_size=BATCH_MIXED, shuffle=False, num_workers=2)

pos_counts = labels.sum(axis=0).clip(min=1)
pos_weight = torch.tensor((N - pos_counts) / pos_counts, dtype=torch.float32).to(DEVICE)
criterion_m = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

model_m = build_model(num_classes=NUM_MIXED_CLASSES, dropout=0.3, pretrained=True).to(DEVICE)
opt_m = optim.AdamW(model_m.parameters(), lr=1e-4, weight_decay=1e-4)
sched_m = OneCycleLR(opt_m, max_lr=5e-4, steps_per_epoch=len(tr_loader), epochs=EPOCHS_MIXED)

MODEL_PATH_MIXED = "models/mixed_model.pth"
best_val_f1 = 0.0

for epoch in range(1, EPOCHS_MIXED + 1):
    if epoch <= 5:
        for p in model_m.features.parameters(): p.requires_grad = False
    else:
        for p in model_m.features.parameters(): p.requires_grad = True

    model_m.train()
    for imgs, lbls in tr_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        opt_m.zero_grad()
        loss = criterion_m(model_m(imgs), lbls)
        loss.backward(); opt_m.step(); sched_m.step()

    model_m.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for imgs, lbls in va_loader:
            probs = torch.sigmoid(model_m(imgs.to(DEVICE))).cpu().numpy()
            all_p.append((probs >= 0.5).astype(int))
            all_l.append(lbls.numpy().astype(int))
    all_p = np.vstack(all_p); all_l = np.vstack(all_l)
    macro_f1_v = f1_score(all_l, all_p, average="macro", zero_division=0)
    print(f"  E{epoch:02d}/{EPOCHS_MIXED} | Val MacroF1: {macro_f1_v:.4f}")
    if macro_f1_v > best_val_f1:
        best_val_f1 = macro_f1_v
        torch.save(model_m.state_dict(), MODEL_PATH_MIXED)
        print(f"       ✅ Saved (best val_f1={best_val_f1:.4f})")

# Test
model_m.load_state_dict(torch.load(MODEL_PATH_MIXED, map_location=DEVICE))
model_m.eval(); all_p, all_l = [], []
with torch.no_grad():
    for imgs, lbls in te_loader:
        probs = torch.sigmoid(model_m(imgs.to(DEVICE))).cpu().numpy()
        all_p.append((probs >= 0.5).astype(int))
        all_l.append(lbls.numpy().astype(int))
all_p = np.vstack(all_p); all_l = np.vstack(all_l)
macro_f1_te = f1_score(all_l, all_p, average="macro", zero_division=0)
hamming = np.mean(all_p != all_l)
print(f"\n✅ MULTI-LABEL TEST: MacroF1={macro_f1_te:.4f}  Hamming Loss={hamming:.4f}  Label Acc={(1-hamming)*100:.1f}%")

# ── [3] COPY MODELS TO DRIVE ─────────────────────────────────────────────────
if os.path.exists(DRIVE_DATA):
    run(f"cp models/wafer_model.pth  {DRIVE_DATA}/wafer_model.pth")
    run(f"cp models/mixed_model.pth  {DRIVE_DATA}/mixed_model.pth")
    print("\n✅ Models saved to Google Drive!")

print("\n" + "=" * 60)
print("  ALL DONE — download models/wafer_model.pth + models/mixed_model.pth")
print("  and replace the models/ folder in your local WaferMap-AI repo")
print("=" * 60)
