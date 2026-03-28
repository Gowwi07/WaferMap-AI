# WaferMapAI – Phase-by-Phase Build Checklist

## Phase 0: Environment Setup & Project Scaffold
- [x] Create project folder structure
- [x] Create requirements.txt
- [x] Create setup_check.py, README.md
- [x] Install all required dependencies (PyTorch, OpenCV, Streamlit, etc.)
- [x] Write Phase 0 Evaluation Report

## Phase 1: Data Gathering & Exploration
- [x] Download WM-811K dataset (LSWMD.pkl, 2GB)
- [x] Understand dataset structure (what a wafer map is, 9 defect classes)
- [x] Write data exploration script (notebooks/explore_data.py)
- [x] Write Phase 1 Evaluation Report

## Phase 2: Model Engineering (EfficientNet-B0 + Transfer Learning)
- [ ] Preprocess dataset (resize, normalize, train/val/test split 70/10/20)
- [ ] Build EfficientNet-B0 model with custom classifier head (9 classes)
- [ ] Train the model
- [ ] Evaluate: accuracy, F1-score per class, confusion matrix
- [ ] Write Phase 2 Evaluation Report

## Phase 3: Grad-CAM Visual Localization
- [ ] Implement Grad-CAM on trained model
- [ ] Generate heatmaps for sample predictions
- [ ] Visualize heatmaps overlaid on wafer maps
- [ ] Write Phase 3 Evaluation Report

## Phase 4: Yield Risk Scoring
- [ ] Implement risk score formula: DefectArea × SeverityWeight × Probability
- [ ] Define severity weights per defect class
- [ ] Integrate into prediction pipeline
- [ ] Write Phase 4 Evaluation Report

## Phase 5: Streamlit App + Deployment
- [ ] Build Streamlit UI (upload wafer image → predict → show heatmap + risk score)
- [ ] Test end-to-end pipeline
- [ ] Write Phase 5 Evaluation Report / Final Demo Notes
