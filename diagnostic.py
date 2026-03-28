import os
import cv2
import numpy as np
import torch
from PIL import Image
import sys
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from src.inference import load_assets, run_inference_with_gradcam
from src.dataset   import CLASS_NAMES

def run_audit():
    print("\n" + "="*60)
    print("  WaferMap AI - Diagnostic Audit (Solution 6)")
    print("="*60)
    
    # 1. Load Assets
    assets = load_assets()
    if not assets:
        print("[Error] Model not found. Please wait for training to finish.")
        return

    # 2. Collect Samples
    sample_dir = os.path.join(ROOT_DIR, "samples_wm811k")
    if not os.path.exists(sample_dir):
        print(f"[Error] Sample directory {sample_dir} not found.")
        return
        
    all_files = [f for f in os.listdir(sample_dir) if f.endswith(".png")]
    y_true, y_pred = [], []
    
    print(f"[Audit] Testing {len(all_files)} samples...")
    
    for fname in all_files:
        # Expected format: wm811k_Donut_123.png
        parts = fname.replace(".png", "").split("_")
        if len(parts) < 2: continue
        true_label = parts[1] # label is the second part
        
        path = os.path.join(sample_dir, fname)
        pil = Image.open(path).convert("RGB")
        
        res = run_inference_with_gradcam(assets, pil)
        pred_label = res.get("predicted_class", "Unknown")
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        
    # 3. Generate Metrics
    print("\n" + "="*60)
    print("  Classification Report (Samples Audit)")
    print("="*60)
    # Filter CLASS_NAMES to only those present in the audit
    present_labels = sorted(list(set(y_true) | set(y_pred)))
    print(classification_report(y_true, y_pred, labels=present_labels))
    
    # 4. Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=present_labels, yticklabels=present_labels, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Diagnostic Confusion Matrix - Solution 6 Audit')
    
    out_path = os.path.join(ROOT_DIR, "reports", "audit_confusion_matrix.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"\n[Audit] Confusion Matrix saved to {out_path}")

if __name__ == "__main__":
    run_audit()
