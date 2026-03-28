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
MODEL_PATH = os.path.join(ROOT_DIR, "models", "mixed_model.pth")

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
    Convert a real wafer photograph (including oval/elliptical microscopy images)
    -> 52x52 int array (0=background, 1=pass, 2=defect/fail).

    Smart Bypass: Already a digital 0/1/2 map -> just resize and remap.
    Photo Path: CLAHE -> ellipse-aware masking -> Otsu + std thresholding -> morph cleanup.
    """
    img_np   = np.array(pil_img.convert("RGB"))
    gray_full = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    unique_vals = np.unique(gray_full)

    # --- FAST PATH: already a digital map ---
    if len(unique_vals) <= 8:
        gray_small = cv2.resize(gray_full, (52, 52), interpolation=cv2.INTER_NEAREST)
        result = np.zeros_like(gray_small, dtype=np.int32)
        result[gray_small > 60]  = 1  # Pass
        result[gray_small > 180] = 2  # Fail
        return result

    # --- PHOTO PATH ---
    h_orig, w_orig = gray_full.shape

    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray_full)

    mask    = np.zeros((h_orig, w_orig), dtype=np.uint8)
    blurred = cv2.GaussianBlur(gray_eq, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
        param1=40, param2=25,
        minRadius=int(min(h_orig, w_orig) * 0.28),
        maxRadius=int(min(h_orig, w_orig) * 0.72)
    )
    if circles is not None:
        cx, cy, cr = np.uint16(np.around(circles[0][0]))
        cv2.ellipse(mask, (int(cx), int(cy)), (int(cr), int(cr * 0.85)),
                    0, 0, 360, 255, -1)
    else:
        _, bw = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if len(largest) >= 5:
                cv2.ellipse(mask, cv2.fitEllipse(largest), 255, -1)
            else:
                cv2.drawContours(mask, [largest], -1, 255, -1)
        else:
            cy2, cx2 = h_orig // 2, w_orig // 2
            cv2.ellipse(mask, (cx2, cy2),
                        (int(w_orig * 0.44), int(h_orig * 0.44)), 0, 0, 360, 255, -1)

    wafer_pixels = gray_eq[mask > 0]
    if len(wafer_pixels) == 0:
        return np.zeros((52, 52), dtype=np.int32)

    mean_v = float(wafer_pixels.mean())
    std_v  = float(wafer_pixels.std())

    inside     = cv2.bitwise_and(gray_eq, gray_eq, mask=mask)
    _, otsu_map = cv2.threshold(inside, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    std_thresh  = max(20, mean_v - 1.8 * std_v)
    std_map     = np.where((gray_eq < std_thresh) & (mask > 0), 255, 0).astype(np.uint8)

    defect_raw   = cv2.bitwise_or(otsu_map, std_map)
    defect_raw   = cv2.bitwise_and(defect_raw, defect_raw, mask=mask)
    defect_clean = cv2.morphologyEx(defect_raw, cv2.MORPH_OPEN,  np.ones((2, 2), np.uint8))
    defect_clean = cv2.morphologyEx(defect_clean, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    mask_s   = cv2.resize(mask,         (52, 52), interpolation=cv2.INTER_NEAREST)
    defect_s = cv2.resize(defect_clean, (52, 52), interpolation=cv2.INTER_NEAREST)

    result = np.zeros((52, 52), dtype=np.int32)
    result[mask_s > 0]   = 1
    result[defect_s > 0] = 2
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
        # Resize background to match heatmap size (224x224) for safe blending
        img_rgb_np = np.array(pil_gray.resize((224, 224)).convert("RGB")) / 255.0
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
