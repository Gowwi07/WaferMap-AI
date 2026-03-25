from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.model import WaferNet


@dataclass
class InferenceAssets:
    model: WaferNet
    labels: list[str]
    device: torch.device
    transform: transforms.Compose


def load_assets(
    model_path: str | Path = "artifacts/wafernet.pt",
    labels_path: str | Path = "artifacts/labels.json",
    img_size: int = 128,
) -> InferenceAssets | None:
    model_path = Path(model_path)
    labels_path = Path(labels_path)
    if not model_path.exists() or not labels_path.exists():
        return None

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaferNet(num_classes=len(labels)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    tfm = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    return InferenceAssets(model=model, labels=labels, device=device, transform=tfm)


def overlay_heatmap(gray_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    gray_rgb = np.stack([gray_img, gray_img, gray_img], axis=-1)
    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heatmap)[..., :3]
    blended = (1 - alpha) * gray_rgb + alpha * heat_rgb
    return np.clip(blended, 0.0, 1.0)


def run_inference_with_gradcam(assets: InferenceAssets, pil: Image.Image) -> dict:
    x = assets.transform(pil).unsqueeze(0).to(assets.device)

    activations = None
    grads = None

    def forward_hook(_module, _inp, out):
        nonlocal activations
        activations = out.detach()

    def backward_hook(_module, _gin, gout):
        nonlocal grads
        grads = gout[0].detach()

    last_conv = assets.model.features[6]
    h1 = last_conv.register_forward_hook(forward_hook)
    h2 = last_conv.register_full_backward_hook(backward_hook)

    logits = assets.model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    pred_idx = int(torch.argmax(probs).item())
    score = logits[0, pred_idx]
    assets.model.zero_grad(set_to_none=True)
    score.backward()

    h1.remove()
    h2.remove()

    if activations is None or grads is None:
        raise RuntimeError("Failed to collect Grad-CAM hooks.")

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
    side = x.shape[-1]
    cam = torch.nn.functional.interpolate(cam, size=(side, side), mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    img_gray = np.array(pil.convert("L").resize((side, side)), dtype=np.float32) / 255.0
    overlay = overlay_heatmap(img_gray, cam, alpha=0.45)

    class_probs = {assets.labels[i]: float(probs[i].item()) for i in range(len(assets.labels))}
    return {
        "predicted_class": assets.labels[pred_idx],
        "confidence": float(probs[pred_idx].item()),
        "class_probabilities": class_probs,
        "gradcam_overlay": overlay,
    }


def overlay_to_png_bytes(overlay: np.ndarray) -> bytes:
    arr = (overlay * 255).astype(np.uint8)
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def overlay_to_base64_png(overlay: np.ndarray) -> str:
    return base64.b64encode(overlay_to_png_bytes(overlay)).decode("utf-8")
