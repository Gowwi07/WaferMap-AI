"""
notebooks/explore_data.py
=========================
Phase 1 - Data Exploration Script

What this script does (in plain English):
1. Loads the WM-811K wafer map dataset from the data/ folder
2. Prints statistics: how many samples exist, how many per defect class
3. Shows sample wafer map images for each defect type
4. Saves the gallery image to reports/phase1_samples.png
5. Saves a class distribution chart to reports/phase1_class_distribution.png

Run this AFTER downloading the dataset:
    python notebooks/explore_data.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (saves to file)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
REPORT_DIR = os.path.join(ROOT_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# ── The 9 defect class names ──────────────────────────────────────────────────
CLASS_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-Full", "Random", "Scratch", "None"
]

# ── Color palette for each class ─────────────────────────────────────────────
CLASS_COLORS = [
    "#e74c3c",  # Center    - red
    "#8e44ad",  # Donut     - purple
    "#2980b9",  # Edge-Loc  - blue
    "#27ae60",  # Edge-Ring - green
    "#f39c12",  # Loc       - orange
    "#c0392b",  # Near-Full - dark red
    "#16a085",  # Random    - teal
    "#d35400",  # Scratch   - burnt orange
    "#95a5a6",  # None      - grey
]

# ── Load dataset ──────────────────────────────────────────────────────────────
def load_dataset():
    """
    The WM-811K dataset is distributed as a pickle (.pkl) file.
    A pickle file is Python's way of saving any object to disk.
    The dataset is a Pandas DataFrame with columns:
      - waferMap : 2D numpy array of 0s (background), 1s (good die), 2s (defect die)
      - failureType : the defect label (may be a list or string)
      - trainTestLabel : 'Training' or 'Test'
    """
    pkl_path = os.path.join(DATA_DIR, "LSWMD.pkl")
    csv_path = os.path.join(DATA_DIR, "WM811K.csv")

    if os.path.exists(pkl_path):
        print(f"[Load] Reading {pkl_path} ...")
        df = pd.read_pickle(pkl_path)
        print(f"[Load] Loaded {len(df):,} records.")
        return df, "pkl"
    elif os.path.exists(csv_path):
        print(f"[Load] Reading {csv_path} ...")
        df = pd.read_csv(csv_path)
        print(f"[Load] Loaded {len(df):,} records.")
        return df, "csv"
    else:
        print("[ERROR] Dataset not found!")
        print("        Expected file: data/LSWMD.pkl")
        print("        Please download the WM-811K dataset from Kaggle.")
        sys.exit(1)


def normalize_label(label):
    """
    The dataset stores labels in different formats depending on version:
    - As a list:  [['Edge-Ring']]
    - As a string: 'Edge-Ring'
    - As empty:   [] or nan
    This function normalizes all of them to a plain string.
    """
    if isinstance(label, (list, np.ndarray)):
        flat = np.array(label).flatten()
        if len(flat) == 0:
            return "none"
        return str(flat[0]).lower().strip()
    if pd.isna(label):
        return "unlabeled"
    return str(label).lower().strip()


def preprocess_df(df):
    """Clean the dataframe and add a 'label' column with normalized class names."""
    print("[Prep] Normalizing labels...")

    # Find the failure type column (may vary by dataset version)
    col = None
    for candidate in ["failureType", "failure_type", "label"]:
        if candidate in df.columns:
            col = candidate
            break
    if col is None:
        print(f"[Prep] Available columns: {list(df.columns)}")
        raise ValueError("Could not find label column in dataset.")

    df = df.copy()
    df["label_raw"] = df[col].apply(normalize_label)

    # Map raw labels to our canonical class names
    label_map = {
        "center":    "Center",
        "donut":     "Donut",
        "edge-loc":  "Edge-Loc",
        "edge-ring": "Edge-Ring",
        "loc":       "Loc",
        "near-full": "Near-Full",
        "random":    "Random",
        "scratch":   "Scratch",
        "none":      "None",
        "unlabeled": "Unlabeled",
    }
    df["label"] = df["label_raw"].map(label_map).fillna("Unlabeled")
    return df


# ── Statistics ────────────────────────────────────────────────────────────────
def print_statistics(df):
    print()
    print("=" * 55)
    print("  WM-811K Dataset Statistics")
    print("=" * 55)
    print(f"  Total records    : {len(df):>10,}")

    labeled   = df[df["label"] != "Unlabeled"]
    unlabeled = df[df["label"] == "Unlabeled"]
    print(f"  Labeled samples  : {len(labeled):>10,}")
    print(f"  Unlabeled samples: {len(unlabeled):>10,}")
    print()
    print("  Class Distribution (labeled only):")
    print("  " + "-" * 45)

    counts = labeled["label"].value_counts()
    total  = len(labeled)
    for cls in CLASS_NAMES:
        count = counts.get(cls, 0)
        pct   = 100.0 * count / total if total else 0
        bar   = "#" * int(pct / 2)
        print(f"  {cls:<12}: {count:>7,}  ({pct:5.1f}%)  {bar}")
    print("=" * 55)
    print()


# ── Chart 1: Class Distribution Bar Chart ─────────────────────────────────────
def plot_class_distribution(df, save_path):
    labeled = df[df["label"] != "Unlabeled"]
    counts  = labeled["label"].value_counts()

    ordered_counts = [counts.get(cls, 0) for cls in CLASS_NAMES]
    pcts = [100.0 * c / sum(ordered_counts) for c in ordered_counts]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    bars = ax.bar(CLASS_NAMES, ordered_counts, color=CLASS_COLORS, edgecolor="#0f3460", linewidth=1.2)

    # Add count and % labels on top of each bar
    for bar, count, pct in zip(bars, ordered_counts, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{count:,}\n({pct:.1f}%)",
            ha="center", va="bottom",
            color="white", fontsize=8.5, fontweight="bold"
        )

    ax.set_title("WM-811K: Class Distribution", color="white", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Number of Samples", color="#aaaaaa", fontsize=11)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#0f3460")
    ax.yaxis.label.set_color("#aaaaaa")
    for label in ax.get_xticklabels(): label.set_color("white")
    for label in ax.get_yticklabels(): label.set_color("#aaaaaa")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Save] Class distribution chart --> {save_path}")


# ── Chart 2: Wafer Map Sample Gallery ─────────────────────────────────────────
def plot_sample_gallery(df, save_path, samples_per_class=3):
    """
    Shows `samples_per_class` example wafer maps for each of the 9 defect types.
    Each wafer map is a 2D grid where:
      0 = background (no die)
      1 = good die (the chip passed quality test)
      2 = defective die (something went wrong here)
    """
    labeled = df[df["label"] != "Unlabeled"]
    n_cols  = samples_per_class
    n_rows  = len(CLASS_NAMES)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(
        "WM-811K Wafer Map Gallery – All 9 Defect Classes",
        color="white", fontsize=14, fontweight="bold", y=1.01
    )

    # Color map: 0=black (background), 1=teal (good die), 2=red (defective die)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#1a1a2e", "#00d2ff", "#e74c3c"])

    for row_idx, cls in enumerate(CLASS_NAMES):
        cls_samples = labeled[labeled["label"] == cls]
        chosen = cls_samples.sample(min(n_cols, len(cls_samples)), random_state=42)

        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            ax.set_facecolor("#1a1a2e")
            ax.axis("off")

            if col_idx == 0:
                ax.set_ylabel(
                    cls, color=CLASS_COLORS[row_idx],
                    fontsize=9, fontweight="bold", rotation=0,
                    labelpad=60, va="center"
                )
                ax.yaxis.set_label_position("left")
                ax.yaxis.label.set_visible(True)

            if col_idx < len(chosen):
                wmap = chosen.iloc[col_idx]["waferMap"]
                wmap = np.array(wmap)
                ax.imshow(wmap, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        color="#555", fontsize=10, transform=ax.transAxes)

    # Legend
    legend_elems = [
        mpatches.Patch(color="#1a1a2e", label="Background"),
        mpatches.Patch(color="#00d2ff", label="Good Die"),
        mpatches.Patch(color="#e74c3c", label="Defective Die"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3,
               facecolor="#16213e", edgecolor="#0f3460",
               labelcolor="white", fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Save] Sample gallery          --> {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print()
    print("=" * 55)
    print("  WaferMapAI - Phase 1: Data Exploration")
    print("=" * 55)
    print()

    # Step 1: Load
    df, fmt = load_dataset()
    print(f"[Info] Dataset format: {fmt}")
    print(f"[Info] Columns: {list(df.columns)}")

    # Step 2: Preprocess
    df = preprocess_df(df)

    # Step 3: Print statistics
    print_statistics(df)

    # Step 4: Charts
    print("[Charts] Generating visualizations...")
    plot_class_distribution(df, os.path.join(REPORT_DIR, "phase1_class_distribution.png"))
    plot_sample_gallery(df, os.path.join(REPORT_DIR, "phase1_samples.png"))

    print()
    print("=" * 55)
    print("  Phase 1 COMPLETE!")
    print(f"  Reports saved to: {REPORT_DIR}")
    print("=" * 55)
    print()
