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
MODEL_PATH = os.path.join(ROOT_DIR, "models", "efficientnet_wafer.pth")

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


def run_inference_with_gradcam(assets: InferenceAssets, pil: Image.Image) -> dict:
    """Run a single prediction with Grad-CAM and risk scoring."""
    # Ensure RGB
    pil = pil.convert("RGB")
    img_tensor = assets.transform(pil)
    img_batch = img_tensor.unsqueeze(0).to(assets.device)

    # Grad-CAM inference
    heatmap, output, pred_class_idx = assets.cam(img_batch)

    probs = F.softmax(output, dim=1)[0].detach().cpu().numpy()
    confidence = float(probs[pred_class_idx])
    pred_class_name = CLASS_NAMES[pred_class_idx]

    # Risk score
    risk_score, action = calculate_risk_score(pred_class_name, confidence, heatmap)

    # Overlay
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
