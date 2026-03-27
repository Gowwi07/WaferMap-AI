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
    """Unified auto-routing endpoint: MixedWM38 (multi) vs WM-811K (single)."""
    if ASSETS is None or MIXED_ASSETS is None:
        raise HTTPException(status_code=503, detail="Both models must be trained before predicting.")

    content = await image.read()
    try:
        pil = Image.open(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open image: {e}")

    # 1. Run Multi-Label model first
    multi_result = run_mixed_inference(MIXED_ASSETS, pil)
    
    # 2. Check if it's truly a multi-defect wafer (>= 2 defects)
    if multi_result.get("defect_count", 0) >= 2:
        multi_result["mode"] = "multi"
        return multi_result

    # 3. Otherwise, fall back to the highly accurate Single-Label model
    single_result = run_inference_with_gradcam(ASSETS, pil)
    if "error" in single_result:
        raise HTTPException(status_code=422, detail=single_result["error"])

    return {
        "mode": "single",
        "class": single_result["predicted_class"],
        "confidence": single_result["confidence"],
        "class_probabilities": single_result["class_probabilities"],
        "risk_score": single_result["risk_score"],
        "action": single_result["action"],
        "gradcam_png_base64": overlay_to_base64_png(single_result["gradcam_overlay"]),
    }
