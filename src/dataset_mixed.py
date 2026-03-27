"""
src/dataset_mixed.py
====================
Dataset loader for MixedWM38.npz — multi-label wafer defect classification.

The dataset has:
  arr_0: (38015, 52, 52) int32  — wafer die maps (0=blank, 1=pass, 2=fail)
  arr_1: (38015, 8)      int32  — one-hot multi-label (8 basic defect types C2–C9)

We map the int wafer arrays to a 3-channel float image and resize to 224×224.
"""

import os, sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "MixedWM38.npz")

IMG_SIZE = 224

# The 8 defect types (C2–C9) corresponding to columns 0–7 of arr_1
MIXED_CLASS_NAMES = [
    "Center",       # C2
    "Donut",        # C3
    "Edge-Loc",     # C4
    "Edge-Ring",    # C5
    "Loc",          # C6
    "Near-Full",    # C7
    "Random",       # C8
    "Scratch",      # C9
]
NUM_MIXED_CLASSES = len(MIXED_CLASS_NAMES)  # 8

# ── Image transform ────────────────────────────────────────────────────────────
# The map values are 0, 1, 2 so we scale to [0, 1] and treat as grayscale→RGB
MIXED_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),                       # [0,255] → [0,1]
    transforms.Lambda(lambda t: t.repeat(3, 1, 1)),  # 1-channel → 3-channel
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def wafer_array_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a (52,52) int32 wafer map (values 0,1,2) → grayscale PIL Image."""
    # Map: 0=background(black), 1=pass(gray), 2=fail(white)
    mapping = np.array([0, 127, 255], dtype=np.uint8)
    img_uint8 = mapping[arr.astype(np.int32).clip(0, 2)]
    return Image.fromarray(img_uint8, mode="L")


class MixedWM38Dataset(Dataset):
    """PyTorch Dataset for the MixedWM38 multi-label wafer defect dataset."""

    def __init__(self, arrays, labels, transform=None):
        self.arrays = arrays          # (N, 52, 52) int32
        self.labels = labels.astype(np.float32)   # (N, 8) float
        self.transform = transform or MIXED_TRANSFORM

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, idx):
        pil = wafer_array_to_pil(self.arrays[idx])
        tensor = self.transform(pil)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return tensor, label


def get_mixed_dataloaders(batch_size=64, num_workers=0, val_ratio=0.1, test_ratio=0.2, seed=42):
    """Load MixedWM38.npz and return (train_loader, val_loader, test_loader, pos_weight)."""
    print(f"[MixedDataset] Loading {DATA_PATH}...")
    data = np.load(DATA_PATH)
    arrays = data["arr_0"]   # (38015, 52, 52)
    labels = data["arr_1"]   # (38015, 8)
    N = len(arrays)
    print(f"[MixedDataset] Loaded {N:,} samples with {NUM_MIXED_CLASSES} label types.")

    # Label statistics
    for i, name in enumerate(MIXED_CLASS_NAMES):
        count = labels[:, i].sum()
        print(f"  {name:<12}: {int(count):>6} ({count/N*100:.1f}%)")

    # Split sizes
    n_test = int(N * test_ratio)
    n_val  = int(N * val_ratio)
    n_train = N - n_test - n_val

    full_ds = MixedWM38Dataset(arrays, labels)
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test], generator=gen)

    print(f"[MixedDataset] Split — Train: {n_train:,}  Val: {n_val:,}  Test: {n_test:,}")

    # Compute pos_weight for BCEWithLogitsLoss to handle class imbalance
    # pos_weight[i] = (N - positives_i) / positives_i
    pos_counts = labels.sum(axis=0).clip(min=1)
    neg_counts = N - pos_counts
    pos_weight = torch.tensor(neg_counts / pos_counts, dtype=torch.float32)
    print(f"[MixedDataset] pos_weight: {pos_weight.numpy().round(2)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, pos_weight
