import os
import cv2
import numpy as np
import torch
from PIL import Image
import sys
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from src.inference_mixed import load_mixed_assets, run_mixed_inference
from src.dataset_mixed   import MIXED_CLASS_NAMES

def run_mixed_audit():
    print("\n" + "="*60)
    print("  WaferMap AI - Multi-Label Audit (MixedWM38)")
    print("="*60)
    
    # 1. Load Assets
    assets = load_mixed_assets()
    if not assets:
        print("[Error] Mixed model not found. Please wait for training to finish.")
        return

    # 2. Collect Samples
    sample_dir = os.path.join(ROOT_DIR, "samples_mixed38")
    if not os.path.exists(sample_dir):
        print(f"[Error] Sample directory {sample_dir} not found.")
        return
        
    all_files = [f for f in os.listdir(sample_dir) if f.endswith(".png")]
    y_true, y_pred = [], []
    
    print(f"[Audit] Testing {len(all_files)} mixed samples...")
    
    for fname in all_files:
        # Expected format: mixed_Donut_Scratch_123.png
        parts = fname.replace(".png", "").split("_")
        # Parts are ['mixed', 'Donut', 'Scratch', '123']
        active_true = parts[1:-1] # Everything between 'mixed' and the numeric ID
        
        # Convert true labels to binary vector
        true_vec = [1 if cls in active_true else 0 for cls in MIXED_CLASS_NAMES]
        
        path = os.path.join(sample_dir, fname)
        pil = Image.open(path).convert("RGB")
        
        res = run_mixed_inference(assets, pil)
        # res['detected_defects'] is a list of strings
        pred_names = res.get('detected_defects', [])
        pred_vec = [1 if cls in pred_names else 0 for cls in MIXED_CLASS_NAMES]
        
        y_true.append(true_vec)
        y_pred.append(pred_vec)
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 3. Generate Metrics
    print("\n" + "="*60)
    print("  Multi-Label Classification Report")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=MIXED_CLASS_NAMES, zero_division=0))
    
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"\n[Audit] Macro F1 Score on Samples: {f1:.4f}")

    # 4. Plot Per-Class Accuracy
    plt.figure(figsize=(10, 5))
    class_acc = (y_true == y_pred).mean(axis=0)
    sns.barplot(x=class_acc, y=MIXED_CLASS_NAMES, palette="viridis")
    plt.xlim(0, 1.05)
    plt.title("Per-Class Accuracy on Mixed Samples")
    plt.xlabel("Accuracy (0-1.0)")
    
    out_path = os.path.join(ROOT_DIR, "reports", "mixed_audit_accuracy.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"\n[Audit] Accuracy chart saved to {out_path}")

if __name__ == "__main__":
    run_mixed_audit()
