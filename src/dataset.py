"""
src/dataset.py
==============
Phase 2 - Data Loading & Preprocessing

What this file does (plain English):
- Reads the raw wafer map data from the pickle file
- Converts each wafer map (2D grid of 0s/1s/2s) into an image the model can understand
- Handles the severe class imbalance (96.9% "None" class)
- Splits data into train / validation / test sets
"""

import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from collections import Counter

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
PKL_PATH   = os.path.join(DATA_DIR, "LSWMD.pkl")
IMG_SIZE   = 224   # Full resolution for GPU training (matches EfficientNet pretrain)
MAX_SAMPLES_PER_CLASS = None  # No cap — use ALL available data (GPU can handle it)

# The 9 defect classes (index = class label for the model)
CLASS_NAMES = [
    "Center",    # 0
    "Donut",     # 1
    "Edge-Loc",  # 2
    "Edge-Ring", # 3
    "Loc",       # 4
    "Near-Full", # 5
    "Random",    # 6
    "Scratch",   # 7
    "None",      # 8
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def normalize_label(label):
    """Convert various label formats in the dataset to a clean string."""
    if isinstance(label, (list, np.ndarray)):
        flat = np.array(label).flatten()
        if len(flat) == 0:
            return None
        val = str(flat[0]).strip()
    elif pd.isna(label) if not isinstance(label, (list, np.ndarray)) else False:
        return None
    else:
        val = str(label).strip()

    mapping = {
        "Center":   "Center",
        "Donut":    "Donut",
        "Edge-Loc": "Edge-Loc",
        "Edge-Ring":"Edge-Ring",
        "Loc":      "Loc",
        "Near-full":"Near-Full",
        "Near-Full":"Near-Full",
        "Random":   "Random",
        "Scratch":  "Scratch",
        "none":     "None",
        "None":     "None",
    }
    return mapping.get(val, None)


def wafer_to_image(wmap):
    """
    Convert a raw wafer map array into a PIL image.

    The wafer map is a 2D numpy array with values:
      0 = background (no die)
      1 = good die (pass)
      2 = defective die (fail)

    We convert this to a grayscale image where:
      0 -> 0   (black background)
      1 -> 127 (grey = good die)
      2 -> 255 (white/bright = defective die)

    Then replicate to 3 channels (RGB) for EfficientNet.
    """
    wmap = np.array(wmap, dtype=np.uint8)
    # Map: 0->0, 1->127, 2->255
    img_arr = np.zeros_like(wmap, dtype=np.uint8)
    img_arr[wmap == 1] = 127
    img_arr[wmap == 2] = 255
    # Convert to PIL, resize, convert to RGB
    img = Image.fromarray(img_arr, mode="L")         # L = grayscale
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    img = img.convert("RGB")                          # 3 channels
    return img


def load_raw_dataframe():
    """Load and preprocess the WM-811K dataframe."""
    print(f"[Dataset] Loading {PKL_PATH}...")
    df = pd.read_pickle(PKL_PATH)
    print(f"[Dataset] Loaded {len(df):,} records.")

    # Normalize labels
    col = "failureType"
    df["label"] = df[col].apply(normalize_label)
    df = df.dropna(subset=["label"])

    # Use all available data (GPU mode — no per-class cap).
    # WeightedRandomSampler in the DataLoader handles class imbalance.
    sampled_parts = []
    for cls in CLASS_NAMES:
        cls_df = df[df["label"] == cls]
        if MAX_SAMPLES_PER_CLASS is not None:
            n = min(len(cls_df), MAX_SAMPLES_PER_CLASS)
            sampled_parts.append(cls_df.sample(n=n, random_state=42))
        else:
            sampled_parts.append(cls_df)

    df_balanced = pd.concat(sampled_parts).sample(frac=1, random_state=42).reset_index(drop=True)

    counts = df_balanced["label"].value_counts()
    print(f"[Dataset] GPU full dataset: {len(df_balanced):,} total samples")
    for cls in CLASS_NAMES:
        print(f"           {cls:<12}: {counts.get(cls,0):>6}")
    return df_balanced



def split_dataframe(df, train_ratio=0.70, val_ratio=0.10):
    """Split into train / validation / test sets."""
    n = len(df)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    train_df = df.iloc[:n_train]
    val_df   = df.iloc[n_train:n_train + n_val]
    test_df  = df.iloc[n_train + n_val:]
    print(f"[Dataset] Split — Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
    return train_df, val_df, test_df


# ── Image transforms ──────────────────────────────────────────────────────────
# Training: augment with flips & rotation (makes model more robust)
# Val/Test: only normalize (no randomness — we want consistent evaluation)

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    # ImageNet mean/std (used because EfficientNet was pretrained on ImageNet)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ── Dataset class ─────────────────────────────────────────────────────────────
class WaferMapDataset(Dataset):
    """
    PyTorch Dataset for the WM-811K wafer maps.

    A Dataset is like a container. PyTorch asks it for items by index:
      dataset[0]  -> (image_tensor, label_number)
    The DataLoader wraps the Dataset to serve batches to the model during training.
    """

    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        wmap  = row["waferMap"]
        label = row["label"]

        # Convert wafer map to PIL Image
        img = wafer_to_image(wmap)

        # Apply transforms (resize, normalize, augment)
        if self.transform:
            img = self.transform(img)

        label_idx = CLASS_TO_IDX[label]
        return img, label_idx


def get_class_weights(df):
    """
    Compute inverse-frequency class weights.
    Rare classes get higher weight so the model pays more attention to them.
    """
    counts = Counter(df["label"].tolist())
    total  = sum(counts.values())
    weights = []
    for cls in CLASS_NAMES:
        c = counts.get(cls, 1)
        weights.append(total / (NUM_CLASSES * c))
    return torch.FloatTensor(weights)


def get_dataloaders(batch_size=32, num_workers=0):
    """
    Build train / val / test DataLoaders.

    DataLoader: handles batching, shuffling, and parallel loading.
    batch_size: how many images to process at once (32 is standard)
    num_workers: parallel data loading processes (0 = main thread, safe for Windows)
    """
    df = load_raw_dataframe()
    train_df, val_df, test_df = split_dataframe(df)

    train_ds = WaferMapDataset(train_df, transform=TRAIN_TRANSFORM)
    val_ds   = WaferMapDataset(val_df,   transform=EVAL_TRANSFORM)
    test_ds  = WaferMapDataset(test_df,  transform=EVAL_TRANSFORM)

    # Weighted sampler: oversample rare classes during training
    cw = get_class_weights(train_df)
    sample_weights = [cw[CLASS_TO_IDX[lbl]] for lbl in train_df["label"]]
    sampler = WeightedRandomSampler(sample_weights, len(train_df), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    class_weights = get_class_weights(train_df)

    print(f"[Dataset] Train batches: {len(train_loader)}")
    print(f"[Dataset] Val   batches: {len(val_loader)}")
    print(f"[Dataset] Test  batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, class_weights


def unnormalize_image(tensor):
    """Convert a normalized PyTorch tensor back to a displayable RGB numpy array (0-1)."""
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    
    img = tensor.cpu().numpy()
    img = (img * std) + mean
    img = np.clip(img, 0, 1)
    # Convert from (C, H, W) to (H, W, C)
    img = np.transpose(img, (1, 2, 0))
    return img


if __name__ == "__main__":
    # Quick test
    train_l, val_l, test_l, cw = get_dataloaders(batch_size=8)
    batch_imgs, batch_labels = next(iter(train_l))
    print(f"[Test] Batch shape: {batch_imgs.shape}")   # [8, 3, 224, 224]
    print(f"[Test] Labels:      {batch_labels}")
    print(f"[Test] Class weights: {cw}")
    print("[Test] dataset.py OK")
