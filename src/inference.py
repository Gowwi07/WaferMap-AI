"""
src/inference.py
================
Inference utilities for the FastAPI backend (app.py).

Loads the trained EfficientNet-B0 model, runs predictions,
and generates Grad-CAM overlays compatible with the existing
src/model.py and src/dataset.py conventions.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.dataset import CLASS_NAMES, NUM_CLASSES, IMG_SIZE
from src.model import build_model
from src.gradcam import GradCAM, overlay_heatmap
from src.risk_score import calculate_risk_score

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "wafer_model.pth")

# Transform must match what was used in training (dataset.py EVAL_TRANSFORM)
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


@dataclass
class InferenceAssets:
    model: torch.nn.Module
    cam: GradCAM
    device: torch.device
    transform: transforms.Compose


def load_assets() -> InferenceAssets | None:
    """Load the trained model + Grad-CAM once at startup."""
    if not os.path.exists(MODEL_PATH):
        print(f"[Inference] Model not found at {MODEL_PATH}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Target the last feature block for Grad-CAM
    target_layer = model.features[-1]
    cam = GradCAM(model, target_layer)

    print(f"[Inference] Model loaded from {MODEL_PATH} on {device}")
    return InferenceAssets(model=model, cam=cam, device=device, transform=EVAL_TRANSFORM)


def unnormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized image tensor back to 0-1 RGB numpy array (H, W, C)."""
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = tensor.cpu().numpy()
    img = (img * std) + mean
    img = np.clip(img, 0, 1)
    return np.transpose(img, (1, 2, 0))  # (C,H,W) -> (H,W,C)


def is_wafer_map(pil_img: Image.Image) -> bool:
    """Heuristic: accept wafer maps AND real wafer photography.
    Deliberately permissive — refuses only clearly invalid inputs.
    """
    w, h = pil_img.size
    aspect = w / float(h)

    # Reject very extreme aspect ratios (panoramas, A4 pages)
    if aspect < 0.35 or aspect > 2.8:
        return False

    gray = np.array(pil_img.convert("L"))

    # Reject completely blank / single-colour images
    counts = np.bincount(gray.flatten())
    if len(counts) > 0:
        most_common_pct = counts.max() / counts.sum()
        if most_common_pct > 0.98:
            return False

    return True

def photo_to_wafer_array(pil_img: Image.Image) -> np.ndarray:
    """
    Convert a real wafer photograph (including oval/elliptical microscopy images)
    -> 52x52 int array (0=background, 1=pass, 2=defect/fail).

    Smart Bypass: If already a digital 0/1/2 map, just resize and remap.
    Photo Path: CLAHE contrast enhancement -> ellipse-aware masking ->
                Otsu + adaptive thresholding -> morphological cleanup.
    """
    img_np = np.array(pil_img.convert("RGB"))
    gray_full = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    unique_vals = np.unique(gray_full)

    # --- FAST PATH: already a digital map (very few distinct intensities) ---
    if len(unique_vals) <= 8:
        gray_small = cv2.resize(gray_full, (52, 52), interpolation=cv2.INTER_NEAREST)
        result = np.zeros_like(gray_small, dtype=np.int32)
        result[gray_small > 50]  = 1  # PASS / die
        result[gray_small > 180] = 2  # FAIL / defect
        return result

    # --- PHOTO PATH: real microscopy / SEM image ---
    h_orig, w_orig = gray_full.shape

    # 1. CLAHE for uniform illumination
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray_full)

    # 2. Build wafer boundary mask
    #    Try HoughCircles first; fall back to fitted ellipse on Canny edges
    mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
    blurred = cv2.GaussianBlur(gray_eq, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
        param1=40, param2=25,
        minRadius=int(min(h_orig, w_orig) * 0.28),
        maxRadius=int(min(h_orig, w_orig) * 0.72)
    )
    if circles is not None:
        cx, cy, cr = np.uint16(np.around(circles[0][0]))
        cv2.ellipse(mask, (int(cx), int(cy)), (int(cr), int(cr*0.85)),
                    0, 0, 360, 255, -1)
    else:
        # Otsu threshold to find wafer blob, then fit ellipse
        _, bw = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # The wafer is typically the large bright oval — find largest contour
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if len(largest) >= 5:
                ellipse = cv2.fitEllipse(largest)
                cv2.ellipse(mask, ellipse, 255, -1)
            else:
                cv2.drawContours(mask, [largest], -1, 255, -1)
        else:
            # Last resort: centre crop assuming ~85% fill
            cy2, cx2 = h_orig // 2, w_orig // 2
            cv2.ellipse(mask, (cx2, cy2),
                        (int(w_orig * 0.44), int(h_orig * 0.44)), 0, 0, 360, 255, -1)

    # 3. Detect defects inside the wafer mask
    #    Dark spots on a beige/grey background == defects (value=2)
    #    Strategy: adaptive threshold to find dark clusters
    inside = cv2.bitwise_and(gray_eq, gray_eq, mask=mask)
    wafer_pixels = gray_eq[mask > 0]
    if len(wafer_pixels) == 0:
        return np.zeros((52, 52), dtype=np.int32)

    mean_v = float(wafer_pixels.mean())
    std_v  = float(wafer_pixels.std())

    # Primary: Otsu on just the wafer region
    _, otsu_map = cv2.threshold(inside, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Secondary: fixed std-based threshold (catches subtle defects)
    std_thresh = max(20, mean_v - 1.8 * std_v)
    std_map = np.where((gray_eq < std_thresh) & (mask > 0),
                       255, 0).astype(np.uint8)

    # Combine both defect maps
    defect_raw = cv2.bitwise_or(otsu_map, std_map)
    defect_raw = cv2.bitwise_and(defect_raw, defect_raw, mask=mask)

    # 4. Morphological cleanup — remove single-pixel salt noise
    kernel3 = np.ones((3, 3), np.uint8)
    defect_clean = cv2.morphologyEx(defect_raw, cv2.MORPH_OPEN,  np.ones((2,2), np.uint8))
    defect_clean = cv2.morphologyEx(defect_clean, cv2.MORPH_CLOSE, kernel3)

    # 5. Build 52×52 result array
    mask_s   = cv2.resize(mask,        (52, 52), interpolation=cv2.INTER_NEAREST)
    defect_s = cv2.resize(defect_clean,(52, 52), interpolation=cv2.INTER_NEAREST)

    result = np.zeros((52, 52), dtype=np.int32)
    result[mask_s > 0]   = 1  # all inside-wafer pixels = PASS
    result[defect_s > 0] = 2  # detected dark clusters = DEFECT

    return result

def wafer_array_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert (52,52) array values (0,1,2) -> grayscale PIL for model input."""
    mapping = np.array([0, 127, 255], dtype=np.uint8)
    img_uint8 = mapping[arr.astype(np.int32).clip(0, 2)]
    return Image.fromarray(img_uint8, mode="L")

def run_inference_with_gradcam(assets: InferenceAssets, pil: Image.Image) -> dict:
    """Run a single prediction with Grad-CAM and risk scoring."""
    if not is_wafer_map(pil):
        return {
            "error": "The uploaded image appears to be a natural photograph or invalid. Please upload a valid circular Wafer Map."
        }
        
    # --- CLEANING STEP (Required for real photos) ---
    # 1. Normalize photo to a clean digital map
    wafer_array = photo_to_wafer_array(pil)
    # 2. Convert back to a PIL image the model can process
    pil_cleaned = wafer_array_to_pil(wafer_array).resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    
    img_tensor = assets.transform(pil_cleaned)
    img_batch = img_tensor.unsqueeze(0).to(assets.device)

    # Grad-CAM inference
    heatmap, output, pred_class_idx = assets.cam(img_batch)

    probs = F.softmax(output, dim=1)[0].detach().cpu().numpy()
    confidence = float(probs[pred_class_idx])
    pred_class_name = CLASS_NAMES[pred_class_idx]

    # Risk score
    risk_score, action = calculate_risk_score(pred_class_name, confidence, heatmap)

    # Overlay - show the heatmap on the CLEANED image so it's clear what the AI saw
    img_rgb = unnormalize_image(img_tensor)
    gradcam_overlay = overlay_heatmap(img_rgb, heatmap)

    class_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    return {
        "predicted_class": pred_class_name,
        "confidence": confidence,
        "class_probabilities": class_probs,
        "risk_score": risk_score,
        "action": action,
        "gradcam_overlay": gradcam_overlay,
    }


def overlay_to_png_bytes(overlay: np.ndarray) -> bytes:
    """Convert an overlay numpy array to PNG bytes."""
    if overlay.max() <= 1.0:
        arr = (overlay * 255).astype(np.uint8)
    else:
        arr = overlay.astype(np.uint8)
    # Convert BGR to RGB if needed (overlay_heatmap uses cv2 which is BGR)
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def overlay_to_base64_png(overlay: np.ndarray) -> str:
    """Convert an overlay numpy array to a base64-encoded PNG string."""
    return base64.b64encode(overlay_to_png_bytes(overlay)).decode("utf-8")
