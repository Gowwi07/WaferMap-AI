"""
src/inference_mixed.py
======================
Inference utilities for the multi-label FastAPI endpoint (/predict_multi).

Supports two input modes (Option B):
  1. 52×52 numpy arrays from .npz files
  2. Real wafer photographs (converted to binary map via adaptive thresholding)
"""

from __future__ import annotations
import base64, io, os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from dataclasses import dataclass

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "efficientnet_mixed.pth")

from src.dataset_mixed import MIXED_TRANSFORM, MIXED_CLASS_NAMES, NUM_MIXED_CLASSES, wafer_array_to_pil
from src.model import build_model
from src.gradcam import GradCAM, overlay_heatmap

THRESHOLD = 0.5  # Sigmoid threshold — above this = defect detected


@dataclass
class MixedInferenceAssets:
    model: torch.nn.Module
    cam:   GradCAM
    device: torch.device


def load_mixed_assets() -> MixedInferenceAssets | None:
    """Load the trained multi-label model once at server startup."""
    if not os.path.exists(MODEL_PATH):
        print(f"[InferenceMixed] Model not found at {MODEL_PATH}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(num_classes=NUM_MIXED_CLASSES, pretrained=False).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    target_layer = model.features[-1]
    cam = GradCAM(model, target_layer)

    print(f"[InferenceMixed] Model loaded from {MODEL_PATH} on {device}")
    return MixedInferenceAssets(model=model, cam=cam, device=device)


def photo_to_wafer_array(pil_img: Image.Image) -> np.ndarray:
    """
    Convert a real wafer photograph → 52×52 int array (values 0, 1, 2).

    Pipeline:
      1. Convert to grayscale
      2. Resize to 52×52
      3. Adaptive threshold to segment the circular die region
      4. Map pixel ranges → {0: background, 1: pass-die, 2: fail-die}
    """
    gray = np.array(pil_img.convert("L"))
    gray_small = cv2.resize(gray, (52, 52), interpolation=cv2.INTER_AREA)

    # Otsu threshold to separate wafer circle from background
    _, bg_mask = cv2.threshold(gray_small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Within the wafer circle, use a second threshold to distinguish pass/fail
    mean_val = gray_small[bg_mask > 0].mean() if bg_mask.sum() > 0 else 128.0
    result = np.zeros_like(gray_small, dtype=np.int32)
    result[bg_mask == 0]  = 0  # Background — outside circular wafer area
    result[(bg_mask > 0) & (gray_small >= mean_val)] = 1   # Bright → pass die
    result[(bg_mask > 0) & (gray_small <  mean_val)] = 2   # Dark  → fail die (defect)

    return result


def run_mixed_inference(assets: MixedInferenceAssets, pil: Image.Image) -> dict:
    """
    Full multi-label inference pipeline.
    Works with both real photos and wafer array images.
    """
    pil_rgb = pil.convert("RGB")

    # Convert to 52×52 array representation
    wafer_array = photo_to_wafer_array(pil_rgb)

    # Convert array back to grayscale PIL for the model transform
    pil_gray = wafer_array_to_pil(wafer_array)
    img_tensor = MIXED_TRANSFORM(pil_gray)
    img_batch  = img_tensor.unsqueeze(0).to(assets.device)

    # Run Grad-CAM (using class=0 for the heatmap; we'll summarize across all)
    with torch.no_grad():
        logits = assets.model(img_batch)    # (1, 8) raw logits
        probs  = torch.sigmoid(logits)[0].cpu().numpy()  # (8,) probabilities

    # Multi-label prediction: all classes above threshold
    predicted_indices = [i for i, p in enumerate(probs) if p >= THRESHOLD]
    detected_defects  = [MIXED_CLASS_NAMES[i] for i in predicted_indices]

    # Generate Grad-CAM for the highest-confidence detected defect
    gradcam_overlay_b64 = None
    if predicted_indices:
        best_class = int(np.argmax(probs))
        heatmap, _, _ = assets.cam(img_batch, class_idx=best_class)
        pil_resized = pil_gray.resize((224, 224), Image.Resampling.NEAREST)
        img_rgb_np = np.array(pil_resized.convert("RGB")) / 255.0
        overlay = overlay_heatmap(img_rgb_np, heatmap)
        # Encode overlay as base64
        if overlay.max() <= 1.0:
            overlay_uint8 = (overlay * 255).astype(np.uint8)
        else:
            overlay_uint8 = overlay.astype(np.uint8)
        overlay_pil = Image.fromarray(overlay_uint8)
        buf = io.BytesIO()
        overlay_pil.save(buf, format="PNG")
        gradcam_overlay_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "detected_defects":    detected_defects,
        "defect_count":        len(detected_defects),
        "is_clean":            len(detected_defects) == 0,
        "confidence_per_class": {
            MIXED_CLASS_NAMES[i]: round(float(probs[i]), 4)
            for i in range(len(MIXED_CLASS_NAMES))
        },
        "gradcam_png_base64": gradcam_overlay_b64,
    }
