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
    """Unified auto-detect router: Single-Label (WM-811K) prioritized as gatekeeper."""
    if ASSETS is None or MIXED_ASSETS is None:
        raise HTTPException(status_code=503, detail="AI models initializing...")

    content = await image.read()
    try:
        pil = Image.open(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Couln't open image: {e}")

    # 1. Run Single-Label model FIRST (Expert in 'None/Clean' and real photos)
    single_result = run_inference_with_gradcam(ASSETS, pil)
    if "error" in single_result:
        raise HTTPException(status_code=422, detail=single_result["error"])

    # 2. If Single-Label model says "None" (Clean), return immediately. 
    # This prevents the Multi-Label model from falsely tagging clean wafers.
    if single_result["predicted_class"] == "None":
        single_result["mode"] = "single"
        return single_result

    # 3. If Single-Label model finds a defect, run Multi-Label (MixedWM38) as a second opinion.
    # It checks if there are other defects hidden in the same image.
    multi_result = run_mixed_inference(MIXED_ASSETS, pil)
    
    # Trust Multi-Label result ONLY if it finds multiple defects (>= 2).
    # If it only finds 1, we stick with the highly-accurate Single-Label model.
    if multi_result.get("defect_count", 0) >= 2:
        multi_result["mode"] = "multi"
        return multi_result

    # Default to Single Result
    single_result["mode"] = "single"
    return single_result
