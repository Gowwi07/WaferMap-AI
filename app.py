from __future__ import annotations

import io

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from src.inference import load_assets, overlay_to_base64_png, run_inference_with_gradcam


app = FastAPI(title="Wafer Map AI API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ASSETS = load_assets()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": ASSETS is not None}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if ASSETS is None:
        return {"error": "Model not found. Train first using src/train.py"}

    content = await image.read()
    pil = Image.open(io.BytesIO(content))
    result = run_inference_with_gradcam(ASSETS, pil)
    return {
        "class": result["predicted_class"],
        "confidence": result["confidence"],
        "class_probabilities": result["class_probabilities"],
        "gradcam_png_base64": overlay_to_base64_png(result["gradcam_overlay"]),
    }
