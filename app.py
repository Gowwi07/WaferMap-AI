from __future__ import annotations

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from src.inference import load_assets, overlay_to_base64_png, run_inference_with_gradcam
from src.inference_mixed import load_mixed_assets, run_mixed_inference


app = FastAPI(title="Wafer Map AI API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load both models at startup
ASSETS       = load_assets()         # Single-label  (WM-811K, 9 classes)
MIXED_ASSETS = load_mixed_assets()   # Multi-label   (MixedWM38, 8 classes)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "single_label_model": ASSETS is not None,
        "multi_label_model":  MIXED_ASSETS is not None,
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """Single-label defect classification (WM-811K, 9 classes)."""
    if ASSETS is None:
        raise HTTPException(status_code=503, detail="Single-label model not found. Run: python src/train.py")

    content = await image.read()
    try:
        pil = Image.open(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open image: {e}")

    result = run_inference_with_gradcam(ASSETS, pil)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return {
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "class_probabilities": result["class_probabilities"],
        "risk_score": result["risk_score"],
        "action": result["action"],
        "gradcam_png_base64": overlay_to_base64_png(result["gradcam_overlay"]),
    }


@app.post("/predict_multi")
async def predict_multi(image: UploadFile = File(...)):
    """Multi-label defect detection (MixedWM38, 8 defect types simultaneously)."""
    if MIXED_ASSETS is None:
        raise HTTPException(
            status_code=503,
            detail="Multi-label model not found. Run: python src/train_mixed.py"
        )

    content = await image.read()
    try:
        pil = Image.open(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open image: {e}")

    result = run_mixed_inference(MIXED_ASSETS, pil)
    return result
