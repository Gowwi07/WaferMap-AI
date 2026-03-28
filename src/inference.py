"""
src/inference.py
================
Inference utilities for the FastAPI backend (app.py).

Critical fix: training uses Image.NEAREST resize from the raw array directly.
We must match this — do NOT downsample to 52×52 first. Instead:
  real photo → full-res binary mask (0/1/2 colours at original size) → PIL → NEAREST resize 224 → model

This matches dataset.py's wafer_to_image() exactly:
  img_arr[wmap == 1] = 127
  img_arr[wmap == 2] = 255
  img.resize((224, 224), Image.NEAREST)
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

# Must match dataset.py EVAL_TRANSFORM — NO Resize here because wafer_to_image already sizes to 224
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),   # safety net if image is already 224
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
    return np.transpose(img, (1, 2, 0))


def is_wafer_map(pil_img: Image.Image) -> bool:
    """Accept wafer maps AND real wafer photography. Refuses obviously invalid images."""
    w, h = pil_img.size
    aspect = w / float(h)
    if aspect < 0.3 or aspect > 3.0:
        return False
    gray = np.array(pil_img.convert("L"))
    counts = np.bincount(gray.flatten())
    if len(counts) > 0 and counts.max() / counts.sum() > 0.99:
        return False
    return True


def _is_digital_map(gray: np.ndarray) -> bool:
    """Return True if the image is already a digital 0/1/2 map (≤ 10 distinct values)."""
    return len(np.unique(gray)) <= 10


def photo_to_model_input(pil_img: Image.Image) -> Image.Image:
    """
    Convert ANY wafer image (digital map OR real photo) into the exact PIL format
    the model expects:  grayscale-encoded RGB image sized IMG_SIZE × IMG_SIZE
    using NEAREST interpolation, matching dataset.py wafer_to_image().

    Values in the resulting image:
      Black  (0)   = background (outside wafer ring)
      Grey (127)   = good dies
      White (255)  = defective dies
    """
    img_np = np.array(pil_img.convert("RGB"))
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    h, w   = gray.shape

    # ─── FAST PATH: already a digital map ─────────────────────────────────────
    if _is_digital_map(gray):
        # The stored sample PNGs already have ~3 distinct pixel values.
        # Map them to exactly 0 / 127 / 255 and resize with NEAREST.
        out = np.zeros_like(gray, dtype=np.uint8)
        out[gray > 50]  = 127   # pass die
        out[gray > 180] = 255   # fail die
        pil_out = Image.fromarray(out, mode="L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        return pil_out.convert("RGB")

    # ─── PHOTO PATH: real microscopy / SEM oval image ─────────────────────────
    # Step 1: CLAHE contrast enhancement
    clahe  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_e = clahe.apply(gray)

    # Step 2: Wafer boundary mask (ellipse-aware)
    mask = np.zeros((h, w), dtype=np.uint8)
    blur = cv2.GaussianBlur(gray_e, (9, 9), 2)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
        param1=40, param2=25,
        minRadius=int(min(h, w) * 0.28),
        maxRadius=int(min(h, w) * 0.72)
    )
    if circles is not None:
        cx, cy, cr = np.uint16(np.around(circles[0][0]))
        # Use 90% of radius to avoid catching the dark border ring as defect
        cv2.ellipse(mask, (int(cx), int(cy)),
                    (int(cr * 0.92), int(cr * 0.78)), 0, 0, 360, 255, -1)
    else:
        # Otsu → largest contour → fitEllipse
        _, bw = cv2.threshold(gray_e, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            largest = max(cnts, key=cv2.contourArea)
            if len(largest) >= 5:
                cv2.ellipse(mask, cv2.fitEllipse(largest), 255, -1)
            else:
                cv2.drawContours(mask, [largest], -1, 255, -1)
        else:
            cv2.ellipse(mask, (w // 2, h // 2),
                        (int(w * 0.44), int(h * 0.44)), 0, 0, 360, 255, -1)

    # Step 3: Defect segmentation at FULL resolution
    wafer_px = gray_e[mask > 0]
    if len(wafer_px) == 0:
        blank = Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8), "L")
        return blank.convert("RGB")

    mean_v = float(wafer_px.mean())
    std_v  = float(wafer_px.std())

    # Primary: Otsu on masked region (handles Donut / Edge-Ring well)
    roi_gray = cv2.bitwise_and(gray_e, gray_e, mask=mask)
    _, otsu  = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Secondary: std-based (catches subtle Near-Full / Random patterns)
    t2      = max(15, mean_v - 2.0 * std_v)
    std_map = np.where((gray_e < t2) & (mask > 0), 255, 0).astype(np.uint8)

    # Combine
    defect_raw = cv2.bitwise_or(otsu, std_map)
    defect_raw = cv2.bitwise_and(defect_raw, defect_raw, mask=mask)

    # Morphological cleanup
    defect = cv2.morphologyEx(defect_raw, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    defect = cv2.morphologyEx(defect,     cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Step 4: Build output image at FULL resolution (matching training convention)
    out = np.zeros((h, w), dtype=np.uint8)
    out[mask > 0]   = 127   # good die (grey)
    out[defect > 0] = 255   # defective die (white)

    # Resize to model input size with NEAREST — same as dataset.py
    pil_out = Image.fromarray(out, mode="L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    return pil_out.convert("RGB")


def run_inference_with_gradcam(assets: InferenceAssets, pil: Image.Image) -> dict:
    """Run a single prediction with Grad-CAM and risk scoring."""
    if not is_wafer_map(pil):
        return {
            "error": "The uploaded image appears to be invalid. Please upload a valid wafer map or wafer photograph."
        }

    # Convert to model-ready image (matches training preprocessing exactly)
    pil_model_input = photo_to_model_input(pil)

    img_tensor = assets.transform(pil_model_input)
    img_batch  = img_tensor.unsqueeze(0).to(assets.device)

    # Grad-CAM inference
    heatmap, output, pred_class_idx = assets.cam(img_batch)

    probs          = F.softmax(output, dim=1)[0].detach().cpu().numpy()
    confidence     = float(probs[pred_class_idx])
    pred_class_name = CLASS_NAMES[pred_class_idx]

    # Risk score
    risk_score, action = calculate_risk_score(pred_class_name, confidence, heatmap)

    # Overlay on cleaned model input (so judges see what the AI saw)
    img_rgb        = unnormalize_image(img_tensor)
    gradcam_overlay = overlay_heatmap(img_rgb, heatmap)

    class_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    return {
        "predicted_class":    pred_class_name,
        "confidence":         confidence,
        "class_probabilities": class_probs,
        "risk_score":         risk_score,
        "action":             action,
        "gradcam_overlay":    gradcam_overlay,
    }


def overlay_to_png_bytes(overlay: np.ndarray) -> bytes:
    if overlay.max() <= 1.0:
        arr = (overlay * 255).astype(np.uint8)
    else:
        arr = overlay.astype(np.uint8)
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def overlay_to_base64_png(overlay: np.ndarray) -> str:
    return base64.b64encode(overlay_to_png_bytes(overlay)).decode("utf-8")
