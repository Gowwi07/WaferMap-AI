"""
src/app.py
==========
Phase 5 - Hackathon Winning UI Upgrade

This Streamlit app has been heavily customized with CSS to look like a modern
Next.js/Vercel application. It features:
1. A hidden default Streamlit menu/footer for a white-label SaaS look.
2. A File Uploader supporting both raw images (.png, .jpg) and dataset .pkl arrays.
3. Real-time inference, Grad-CAM overlays, and Yield Risk Score calculations.
"""

import os, sys
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import pandas as pd
import time
import matplotlib.pyplot as plt
from PIL import Image
import io

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.dataset import CLASS_NAMES, NUM_CLASSES, normalize_label, get_dataloaders, unnormalize_image
from src.model import build_model
from src.gradcam import GradCAM, overlay_heatmap
from src.risk_score import calculate_risk_score
from torchvision import transforms

st.set_page_config(
    page_title="WaferMap AI Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PREMIUM VERCEL-STYLE CSS ---
st.markdown("""
<style>
    /* Hide Streamlit Default UI Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Vercel-like sleek dark mode */
    .stApp {
        background-color: #000000;
        color: #ededed;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #111111;
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 24px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.5);
        margin-bottom: 20px;
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #666;
    }
    
    .metric-value { font-size: 2.5rem; font-weight: 700; color: #ffffff; margin:0;}
    .metric-label { font-size: 0.875rem; color: #888888; text-transform: uppercase; letter-spacing: 1px; margin:0 0 8px 0;}
    .action-header { font-size: 1.5rem; font-weight: 700; margin-bottom: 5px;}
    
    /* Status Colors */
    .status-monitor     { border-top: 4px solid #2ecc71; }
    .status-investigate { border-top: 4px solid #f1c40f; }
    .status-stop        { border-top: 4px solid #e74c3c; }
    
    h1, h2, h3 { color: #ffffff; letter-spacing: -0.02em; }
    
    /* File uploader styling override */
    .stFileUploader > div > div {
        background-color: #111111 !important;
        border: 1px dashed #333333 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading AI Engine...")
def load_app_model():
    """Load model once and cache it in memory."""
    device = torch.device("cpu")
    model_path = os.path.join(ROOT_DIR, "models", "efficientnet_wafer.pth")
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at {model_path}. Please complete Phase 2 first.")
        st.stop()
        
    model = build_model(num_classes=NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    target_layer = model.features[-1]
    cam = GradCAM(model, target_layer)
    return model, cam, device

@st.cache_resource
def load_sample_dataset():
    """Load the test split to provide sample images."""
    _, _, test_loader, _ = get_dataloaders(batch_size=1, num_workers=0)
    return test_loader.dataset

def process_uploaded_image(uploaded_file):
    """Convert an uploaded image file into an EfficientNet-ready Tensor."""
    img = Image.open(uploaded_file).convert("RGB")
    
    # Needs to match Phase 2 dataset.py transform
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(img)
    return tensor, "Uploaded Image"

def main():
    st.title("⚡ WaferMap AI System")
    st.markdown("Automated semiconductor defect classification with **Grad-CAM spatial localization** and **Yield Risk Scoring**.")
    st.markdown("---")
    
    model, cam, device = load_app_model()
    test_dataset = load_sample_dataset()
    
    # --- UI LAYOUT ---
    col_input, col_display = st.columns([1, 2])
    
    img_tensor = None
    true_label_name = None
    
    with col_input:
        st.markdown("### Input Source")
        
        # 1. FILE UPLOAD FEATURE
        uploaded_file = st.file_uploader("Upload Wafer Map (.png, .jpg)", type=["png", "jpg", "jpeg"])
        
        st.markdown("---")
        st.markdown("Or select from the Database:")
        
        # 2. SAMPLE SELECTOR
        selected_class = st.selectbox("Filter Database by True Defect Type", ["Any"] + CLASS_NAMES)
        
        indices = []
        for i in range(len(test_dataset)):
            if selected_class == "Any" or CLASS_NAMES[test_dataset[i][1]] == selected_class:
                indices.append(i)
                if len(indices) > 50: break
                
        selected_idx = st.slider("Select Sample #", 0, len(indices)-1 if len(indices)>0 else 0, 0)
        
        if uploaded_file is not None:
            img_tensor, true_label_name = process_uploaded_image(uploaded_file)
            st.success(f"Loaded: {uploaded_file.name}")
        else:
            actual_idx = indices[selected_idx]
            img_tensor, true_label_idx = test_dataset[actual_idx]
            true_label_name = CLASS_NAMES[true_label_idx]
            st.info(f"Loaded Sample ID: `{actual_idx}`  \nTrue Label: **{true_label_name}**")

        run_btn = st.button("▶️ Analyze Wafer", use_container_width=True, type="primary")

    with col_display:
        if not run_btn and img_tensor is not None:
            st.markdown("### Input Preview")
            img_rgb_display = unnormalize_image(img_tensor)
            st.image(img_rgb_display, width=300, caption="Waiting for analysis...")
            
        elif run_btn and img_tensor is not None:
            with st.spinner("Running EfficientNet-B0 Inference..."):
                time.sleep(0.3) # UI Polish
                
                img_batch = img_tensor.unsqueeze(0).to(device)
                
                # INFERENCE & CAM
                heatmap_map, output, pred_class_idx = cam(img_batch)
                
                probs = F.softmax(output, dim=1)[0].detach().numpy()
                confidence = float(probs[pred_class_idx])
                pred_class_name = CLASS_NAMES[pred_class_idx]
                
                # RISK SCORING
                risk_score, action = calculate_risk_score(pred_class_name, confidence, heatmap_map)
                
                img_rgb_display = unnormalize_image(img_tensor)
                heatmap_overlay = overlay_heatmap(img_rgb_display, heatmap_map)
                
            # --- RENDER RESULTS ---
            c1, c2, c3 = st.columns(3)
            with c1:
                color = "#2ecc71" if pred_class_name == true_label_name else ("#ffffff" if true_label_name == "Uploaded Image" else "#e74c3c")
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">AI Classification</p>
                    <p class="metric-value" style="color:{color}">{pred_class_name}</p>
                    <p style="margin:0; color:#888">Confidence: {confidence*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Yield Risk Score</p>
                    <p class="metric-value">{risk_score} <span style="font-size:1rem;color:#888">/ 100</span></p>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                status_class = "status-monitor" if action == "MONITOR" else "status-investigate" if action == "INVESTIGATE" else "status-stop"
                st.markdown(f"""
                <div class="metric-card {status_class}">
                    <p class="metric-label">Factory Action</p>
                    <p class="action-header">{action}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### Spatial Root-Cause Localization")
            vc1, vc2 = st.columns(2)
            with vc1:
                st.image(img_rgb_display, use_container_width=True, caption=f"Original Input ({true_label_name})")
            with vc2:
                st.image(heatmap_overlay, use_container_width=True, caption=f"Grad-CAM Attention (Pred: {pred_class_name})")

if __name__ == "__main__":
    main()
