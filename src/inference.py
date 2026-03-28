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
    """Heuristic to detect if an image is a wafer map vs a natural photo.
    Kept deliberately loose — better to classify a non-wafer than to refuse a real wafer.
    """
    w, h = pil_img.size
    aspect = w / float(h)

    # Reject very non-square images (e.g. banner screenshots or A4 pages)
    if aspect < 0.5 or aspect > 2.0:
        return False

    gray = np.array(pil_img.convert("L"))

    # Reject images that are 95%+ a single exact pixel value (blank docs)
    counts = np.bincount(gray.flatten())
    if len(counts) > 0:
        most_common_pct = counts.max() / counts.sum()
        if most_common_pct > 0.95:
            return False

    return True

def photo_to_wafer_array(pil_img: Image.Image) -> np.ndarray:
    """
    Convert a real wafer photograph -> 52x52 int array (0, 1, 2).
    Smart Bypass: If the image is already a 0/1/2 digital map, skip processing.
    """
    # 1. Check if it's already a digital map (3-5 unique colors max)
    gray_full = np.array(pil_img.convert("L"))
    unique_vals = np.unique(gray_full)
    
    # If the image has very few unique colors, it's likely a pre-rendered map
    # We just need to map it back to 0, 1, 2 and resize.
    if len(unique_vals) <= 5:
        gray_small = cv2.resize(gray_full, (52, 52), interpolation=cv2.INTER_NEAREST)
        result = np.zeros_like(gray_small, dtype=np.int32)
        # Assuming typical 0=BG, 127=PASS, 255=FAIL mapping
        result[gray_small > 50] = 1 # PASS
        result[gray_small > 200] = 2 # FAIL
        return result

    # 2. Proceed with PHOTO quality cleaning (for real photographs)
    img_np = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    
    blurred = cv2.GaussianBlur(gray_enhanced, (7, 7), 1.5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, 1, minDist=50,
        param1=30, param2=20,
        minRadius=int(min(gray.shape)*0.25), maxRadius=int(min(gray.shape)*0.75)
    )
    
    mask = np.zeros_like(gray)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
        cv2.circle(mask, (x, y), r, 255, -1)
    else:
        h, w = gray.shape
        cv2.circle(mask, (w//2, h//2), int(min(h, w)*0.45), 255, -1)

    gray_small = cv2.resize(gray_enhanced, (52, 52), interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(mask, (52, 52), interpolation=cv2.INTER_NEAREST)
    
    wafer_pixels = gray_small[mask_small > 0]
    result = np.zeros_like(gray_small, dtype=np.int32)
    
    if len(wafer_pixels) > 0:
        mean_val = wafer_pixels.mean()
        std_val = wafer_pixels.std()
        result[mask_small > 0] = 1 
        threshold = mean_val - (1.5 * std_val)
        result[(mask_small > 0) & (gray_small < threshold)] = 2
        
        # Clean up noise (Morphological Opening)
        kernel = np.ones((2, 2), np.uint8)
        defect_mask = (result == 2).astype(np.uint8)
        defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel)
        result[mask_small > 0] = 1 
        result[defect_mask > 0] = 2
        
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
