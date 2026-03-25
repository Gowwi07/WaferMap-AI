# WaferMapAI

An AI-powered wafer defect classification system built for the PSG iTech × SanDisk Hackathon.

## What It Does
- Classifies 9 wafer defect patterns using EfficientNet-B0 (~97% accuracy)
- Localizes defect regions with Grad-CAM heatmaps
- Outputs a Yield Risk Score (0–100) for each wafer

## Project Structure
```
.
├── data/           ← WM-811K dataset
├── models/         ← Saved trained weights
├── notebooks/      ← Exploration & visualization scripts
├── src/
│   ├── dataset.py  ← Data loading and preprocessing
│   ├── model.py    ← EfficientNet-B0 model
│   ├── train.py    ← Training pipeline
│   ├── gradcam.py  ← Grad-CAM implementation
│   ├── risk_score.py ← Yield risk scoring
│   └── app.py      ← Streamlit web app
├── reports/        ← Phase evaluation reports
├── requirements.txt
└── setup_check.py  ← Verify your environment
```

## Phases
| Phase | Description |
|---|---|
| 0 | Environment Setup |
| 1 | Data Gathering & Exploration |
| 2 | Model Training (EfficientNet-B0) |
| 3 | Grad-CAM Visualization |
| 4 | Yield Risk Scoring |
| 5 | Streamlit App |

## Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify setup
python setup_check.py

# 3. Explore data
python notebooks/explore_data.py

# 4. Train model
python src/train.py

# 5. Run Grad-CAM on sample image
python src/gradcam.py --image data/wafer_synth/center/center_0001.png

# 6. Launch app
streamlit run src/app.py

# 7. (Optional) API with embedded Grad-CAM image (base64)
uvicorn app:app --reload
```

Training now also saves:
- `artifacts/confusion_matrix.png`
- `artifacts/classification_report.json` (includes per-class precision/recall/F1)

The Streamlit app now shows:
- uploaded wafer image
- predicted class + confidence
- Grad-CAM heatmap
- class-wise probabilities
- saved training metrics table + confusion matrix

## Team
- Gowtham R (Leader)
- Mohan Kumar G
- PSG Institute of Technology and Applied Research
