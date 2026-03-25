"""
notebooks/visualize_gradcam.py
==============================
Phase 3 - Visualize Grad-CAM Heatmaps

This script loads the trained model from Phase 2 and applies our Grad-CAM 
implementation to one random image from each defect class.

It generates a grid image (reports/phase3_gradcam.png) showing:
[Original Wafer] | [Grad-CAM Heatmap Overlay] 

This helps engineers visually confirm that the AI is looking at the actual 
defect rather than background noise.
"""

import os, sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.dataset import get_dataloaders, CLASS_NAMES, NUM_CLASSES, unnormalize_image
from src.model   import build_model
from src.gradcam import GradCAM, overlay_heatmap

MODELS_DIR  = os.path.join(ROOT_DIR, "models")
REPORT_DIR  = os.path.join(ROOT_DIR, "reports")
MODEL_PATH  = os.path.join(MODELS_DIR, "efficientnet_wafer.pth")
OUTPUT_PATH = os.path.join(REPORT_DIR, "phase3_gradcam.png")
DEVICE      = torch.device("cpu")

def main():
    print(f"==================================================")
    print(f"  Phase 3: Generating Grad-CAM Heatmaps")
    print(f"==================================================")

    # 1. Load data
    # We just need the test loader to grab a few sample images
    print("[1/4] Loading test dataset...")
    _, _, test_loader, _ = get_dataloaders(batch_size=1, num_workers=0)
    dataset = test_loader.dataset

    # Find one sample image for each of the 9 classes
    sample_indices = {}
    for i in range(len(dataset)):
        _, label = dataset[i]
        label_name = CLASS_NAMES[label]
        if label_name not in sample_indices:
            sample_indices[label_name] = i
        if len(sample_indices) == NUM_CLASSES:
            break

    # 2. Load the trained model
    print("[2/4] Loading trained EfficientNet model...")
    model = build_model(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run Phase 2 first!")
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 3. Initialize Grad-CAM
    print("[3/4] Initializing Grad-CAM on layer 'features[-1]'...")
    # EfficientNet-B0 has a 'features' sequential block. 
    # The last element [-1] is the final convolutional layer before pooling.
    target_layer = model.features[-1]
    cam = GradCAM(model, target_layer)

    # 4. Generate visualizations
    print("[4/4] Processing Heatmaps...")
    fig, axes = plt.subplots(NUM_CLASSES, 2, figsize=(8, 3 * NUM_CLASSES))
    fig.patch.set_facecolor("#1a1a2e")

    row = 0
    for cls_name in CLASS_NAMES:
        idx = sample_indices[cls_name]
        img_tensor, true_label = dataset[idx]
        
        # Add batch dimension: (1, C, H, W)
        img_batch = img_tensor.unsqueeze(0).to(DEVICE)

        # Run Grad-CAM
        # We ask Grad-CAM to explain the TRUE label, so we see what features support the correct class
        heatmap_map, output, _ = cam(img_batch, class_idx=true_label)
        
        pred_label_idx = output.argmax(dim=1).item()
        pred_name      = CLASS_NAMES[pred_label_idx]

        # Convert tensor to RGB image for display
        img_rgb = unnormalize_image(img_tensor)

        # Create overlay
        overlaid = overlay_heatmap(img_rgb, heatmap_map)

        # Plot Original
        ax1 = axes[row, 0]
        ax1.imshow(img_rgb)
        ax1.axis("off")
        ax1.set_title(f"Original: {cls_name}", color="white")

        # Plot Grad-CAM
        ax2 = axes[row, 1]
        ax2.imshow(overlaid)
        ax2.axis("off")
        title_color = "#2ecc71" if pred_name == cls_name else "#e74c3c"
        ax2.set_title(f"GradCAM (Pred: {pred_name})", color=title_color)

        row += 1

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

    print(f"\n[DONE] Saved visualizations to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
