"""
src/inference_mixed.py
======================
Inference utilities for the multi-label FastAPI endpoint (/predict_multi).

Critical fix: same as inference.py — do full-res binary segmentation at input
resolution, then resize to 224 via NEAREST, matching dataset_mixed.py training.
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

THRESHOLD = 0.5  # Sigmoid threshold


@dataclass
class MixedInferenceAssets:
    model: torch.nn.Module
    cam:   GradCAM
    device: torch.device


def load_mixed_assets() -> MixedInferenceAssets | None:
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


def _is_digital_map(gray: np.ndarray) -> bool:
    return len(np.unique(gray)) <= 10


def photo_to_model_input_mixed(pil_img: Image.Image) -> Image.Image:
    """
    Convert ANY wafer image into the PIL format the multi-label model expects,
    matching dataset_mixed.py training: grayscale encoded 0/127/255 → RGB 224×224.
    """
    img_np = np.array(pil_img.convert("RGB"))
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    h, w   = gray.shape

    # ─── FAST PATH: already a digital map ─────────────────────────────────────
    if _is_digital_map(gray):
        out = np.zeros_like(gray, dtype=np.uint8)
        out[gray > 60]  = 127
        out[gray > 180] = 255
        pil_out = Image.fromarray(out, "L").resize((224, 224), Image.NEAREST)
        return pil_out.convert("RGB")

    # ─── PHOTO PATH ───────────────────────────────────────────────────────────
    clahe  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_e = clahe.apply(gray)

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
        cv2.ellipse(mask, (int(cx), int(cy)),
                    (int(cr * 0.92), int(cr * 0.78)), 0, 0, 360, 255, -1)
    else:
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

    wafer_px = gray_e[mask > 0]
    if len(wafer_px) == 0:
        blank = Image.fromarray(np.zeros((224, 224), dtype=np.uint8), "L")
        return blank.convert("RGB")

    mean_v = float(wafer_px.mean())
    std_v  = float(wafer_px.std())

    roi = cv2.bitwise_and(gray_e, gray_e, mask=mask)
    _, otsu = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    t2      = max(15, mean_v - 2.0 * std_v)
    std_map = np.where((gray_e < t2) & (mask > 0), 255, 0).astype(np.uint8)

    defect_raw = cv2.bitwise_or(otsu, std_map)
    defect_raw = cv2.bitwise_and(defect_raw, defect_raw, mask=mask)
    defect = cv2.morphologyEx(defect_raw, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    defect = cv2.morphologyEx(defect,     cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    out = np.zeros((h, w), dtype=np.uint8)
    out[mask > 0]   = 127
    out[defect > 0] = 255

    pil_out = Image.fromarray(out, "L").resize((224, 224), Image.NEAREST)
    return pil_out.convert("RGB")


def run_mixed_inference(assets: MixedInferenceAssets, pil: Image.Image) -> dict:
    """Full multi-label inference pipeline."""
    pil_model_input = photo_to_model_input_mixed(pil.convert("RGB"))

    img_tensor = MIXED_TRANSFORM(pil_model_input)
    img_batch  = img_tensor.unsqueeze(0).to(assets.device)

    with torch.no_grad():
        logits = assets.model(img_batch)
        probs  = torch.sigmoid(logits)[0].cpu().numpy()

    predicted_indices = [i for i, p in enumerate(probs) if p >= THRESHOLD]
    detected_defects  = [MIXED_CLASS_NAMES[i] for i in predicted_indices]

    gradcam_overlay_b64 = None
    if predicted_indices:
        best_class = int(np.argmax(probs))
        heatmap, _, _ = assets.cam(img_batch, class_idx=best_class)
        img_rgb_np = np.array(pil_model_input.resize((224, 224))) / 255.0
        overlay = overlay_heatmap(img_rgb_np, heatmap)
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
