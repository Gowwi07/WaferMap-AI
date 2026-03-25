"""
src/app.py
==========
Phase 5 - Streamlit Web Dashboard

This is the final piece of the WaferMapAI project. It provides an interactive
UI for engineers to:
1. Upload a raw wafer map image (or select a sample)
2. Run the AI prediction (EfficientNet classification)
3. View the Grad-CAM heatmap explanation
4. See the Yield Risk Score and recommended action

Run with:
    streamlit run src/app.py
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

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.dataset import CLASS_NAMES, NUM_CLASSES, normalize_label, get_dataloaders, unnormalize_image
from src.model import build_model
from src.gradcam import GradCAM, overlay_heatmap
from src.risk_score import calculate_risk_score

st.set_page_config(
    page_title="WaferMap AI Dashboard",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI aesthetics
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e2f;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border-left: 5px solid #00d2ff;
    }
    .metric-value { font-size: 2.5rem; font-weight: bold; color: white; margin:0;}
    .metric-label { font-size: 1.1rem; color: #a0aec0; text-transform: uppercase; letter-spacing: 1px; margin:0;}
    .action-header { font-size: 1.5rem; font-weight: bold; margin-bottom: 5px;}
    
    .status-monitor     { border-left-color: #2ecc71; }
    .status-investigate { border-left-color: #f1c40f; }
    .status-stop        { border-left-color: #e74c3c; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
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
    
    # Initialize Grad-CAM
    target_layer = model.features[-1]
    cam = GradCAM(model, target_layer)
    return model, cam, device

@st.cache_resource
def load_sample_dataset():
    """Load the test split to provide sample images."""
    _, _, test_loader, _ = get_dataloaders(batch_size=1, num_workers=0)
    return test_loader.dataset

def main():
    st.title("🖥️ WaferMap AI: Automated Defect Classification")
    st.markdown("An end-to-end AI pipeline that classifies wafer defects, highlights anomalies via **Grad-CAM**, and assigns a real-time **Yield Risk Score**.")
    
    # --- Sidebar ---
    st.sidebar.header("Control Panel")
    
    # Loading resources
    with st.spinner("Loading AI Engine..."):
        model, cam, device = load_app_model()
        test_dataset = load_sample_dataset()
        
    st.sidebar.success("Model Engine Ready")
    st.sidebar.markdown("---")
    
    # Input Selection
    st.sidebar.subheader("1. Select Input")
    input_method = st.sidebar.radio("Input Method", ["Use Sample Database"])
    
    img_tensor = None
    true_label_name = None
    
    # Let user pick a class, then pick a random image from that class
    selected_class = st.sidebar.selectbox("Filter by True Defect Type", ["Any"] + CLASS_NAMES)
    
    # We find indices match
    indices = []
    for i in range(len(test_dataset)):
        if selected_class == "Any" or CLASS_NAMES[test_dataset[i][1]] == selected_class:
            indices.append(i)
            if len(indices) > 50: # Cap so it runs fast
                break
                
    selected_idx = st.sidebar.slider("Select Sample #", 0, len(indices)-1 if len(indices)>0 else 0, 0)
    actual_idx = indices[selected_idx]
    
    img_tensor, true_label_idx = test_dataset[actual_idx]
    true_label_name = CLASS_NAMES[true_label_idx]
    st.sidebar.info(f"Loaded database sample `{actual_idx}`\n\nTrue Label: **{true_label_name}**")

    run_btn = st.sidebar.button("▶️ Run AI Analysis", use_container_width=True, type="primary")

    # --- Main Area ---
    if run_btn:
        with st.spinner("Executing Neural Network Pipeline..."):
            time.sleep(0.5) # Slight UX delay
            
            # Add batch dimension
            img_batch = img_tensor.unsqueeze(0).to(device)
            
            # --- MODEL INFERENCE & GRAD-CAM ---
            # Forward pass through Grad-CAM
            heatmap_map, output, pred_class_idx = cam(img_batch)
            
            # Softmax probabilities
            probs = F.softmax(output, dim=1)[0].detach().numpy()
            confidence = float(probs[pred_class_idx])
            pred_class_name = CLASS_NAMES[pred_class_idx]
            
            # --- RISK SCORING ---
            risk_score, action = calculate_risk_score(pred_class_name, confidence, heatmap_map)
            
            # --- IMAGE PREPARATION ---
            img_rgb_display = unnormalize_image(img_tensor)
            heatmap_overlay = overlay_heatmap(img_rgb_display, heatmap_map)

        # --- TOP METRICS ROW ---
        col1, col2, col3 = st.columns(3)
        
        with col1:
            color = "#2ecc71" if pred_class_name == true_label_name else "#e74c3c"
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">AI Classification</p>
                <p class="metric-value" style="color:{color}">{pred_class_name}</p>
                <p style="margin:0; color:#aaa">Confidence: {confidence*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Yield Risk Score</p>
                <p class="metric-value">{risk_score} / 100</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            status_class = "status-monitor" if action == "MONITOR" else "status-investigate" if action == "INVESTIGATE" else "status-stop"
            st.markdown(f"""
            <div class="metric-card {status_class}">
                <p class="metric-label">Recommended Action</p>
                <p class="action-header">{action}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # --- VISUALIZATION ROW ---
        st.subheader("Visual Explanations")
        vc1, vc2, vc3 = st.columns([1,1,1])
        
        with vc1:
            st.markdown("**Original Wafer Map** (Input)")
            st.image(img_rgb_display, use_container_width=True)
            
        with vc2:
            st.markdown("**Grad-CAM Attention Map**")
            # Just the raw heatmap using matplotlib
            fig, ax = plt.subplots(figsize=(4,4))
            fig.patch.set_facecolor("#1e1e2f")
            ax.imshow(heatmap_map, cmap="jet")
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)
            
        with vc3:
            st.markdown("**Overlay Explanation**")
            st.image(heatmap_overlay, use_container_width=True)

        # --- DETAILED PROBABILITIES ---
        st.markdown("---")
        with st.expander("Show AI Raw Probabilities"):
            probs_df = pd.DataFrame({
                "Defect Class": CLASS_NAMES,
                "Probability": probs
            }).sort_values(by="Probability", ascending=False)
            
            st.bar_chart(probs_df.set_index("Defect Class"))
            
    else:
        # Default state
        st.info("👈 Select an input from the sidebar and click **Run AI Analysis** to start.")
        st.markdown("""
        ### How it works:
        1. **Input:** The dashboard extracts a silicon wafer map image from our phase 1 database.
        2. **Classification:** It feeds it into the trained EfficientNet-B0 CNN (Phase 2).
        3. **Explainability:** Grad-CAM traces gradients backwards to visualize *where* the CNN found the defect (Phase 3).
        4. **Decision:** The final Yield Risk Score evaluates the severity and surface area to recommend factory line actions (Phase 4).
        """)

if __name__ == "__main__":
    main()
