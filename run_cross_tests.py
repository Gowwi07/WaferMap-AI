
import os
import torch
import numpy as np
from PIL import Image
import sys

# Add current dir to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from src.inference import load_assets, run_inference_with_gradcam
from src.inference_mixed import load_mixed_assets, run_mixed_inference, wafer_array_to_pil

def run_tests():
    print("--- WaferMap AI: Cross-Dataset Model Testing ---")
    
    # 1. Load Assets
    print("\n[Init] Loading Single-Label Model (WM-811K)...")
    single_assets = load_assets()
    print("[Init] Loading Multi-Label Model (MixedWM38)...")
    mixed_assets = load_mixed_assets()
    
    if not single_assets or not mixed_assets:
        print("[Error] Models not found. Please ensure training is complete.")
        return

    # 2. Prepare Test Images
    test_cases = []
    
    # CASE A: WM-811K Samples (Single Defect)
    wm_samples = ["center.png", "scratch.png", "edge_ring.png"]
    for s in wm_samples:
        path = os.path.join(ROOT_DIR, "frontend", "samples", s)
        if os.path.exists(path):
            test_cases.append({
                "name": f"WM-811K: {s}",
                "img": Image.open(path).convert("RGB"),
                "source": "WM-811K"
            })

    # CASE B: MixedWM38 Samples
    print("[Init] Extracting samples from MixedWM38.npz...")
    try:
        data = np.load(os.path.join(ROOT_DIR, "data", "MixedWM38.npz"))
        arrays = data["arr_0"]
        labels = data["arr_1"]
        
        # Find a multi-defect sample (sum of labels > 1)
        multi_idx = np.where(labels.sum(axis=1) > 1)[0][0]
        # Find a single-defect sample (sum of labels == 1)
        single_idx = np.where(labels.sum(axis=1) == 1)[0][0]
        
        test_cases.append({
            "name": "MixedWM38: Multi-Defect",
            "img": wafer_array_to_pil(arrays[multi_idx]).convert("RGB"),
            "source": "MixedWM38",
            "ground_truth": [label_name(i) for i, v in enumerate(labels[multi_idx]) if v == 1]
        })
        test_cases.append({
            "name": "MixedWM38: Single-Defect",
            "img": wafer_array_to_pil(arrays[single_idx]).convert("RGB"),
            "source": "MixedWM38",
            "ground_truth": [label_name(i) for i, v in enumerate(labels[single_idx]) if v == 1]
        })
    except Exception as e:
        print(f"[Warning] Could not load MixedWM38 samples: {e}")

    # 3. Run Inference
    print("\n" + "="*110)
    print(f"{'Test Image':<30} | {'Ground Truth':<25} | {'Single-Label Result':<20} | {'Multi-Label Detections'}")
    print("-" * 110)
    
    for case in test_cases:
        img = case["img"]
        gt = ", ".join(case.get("ground_truth", ["N/A"]))
        
        # Single-Label Prediction
        single_res = run_inference_with_gradcam(single_assets, img)
        pred_single = single_res.get("predicted_class", "Error")
        if "error" in single_res: pred_single = "Invalid Format"
        
        # Multi-Label Prediction
        mixed_res = run_mixed_inference(mixed_assets, img)
        detections = ", ".join(mixed_res.get("detected_defects", []))
        if not detections: detections = "None"
        
        print(f"{case['name']:<30} | {gt:<25} | {pred_single:<20} | {detections}")

def label_name(idx):
    names = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-Full", "Random", "Scratch"]
    return names[idx]

if __name__ == "__main__":
    run_tests()
