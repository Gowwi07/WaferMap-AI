# Phase 3: Grad-CAM Visual Localization

**Goal:** Make the AI "show its work" by visually highlighting exactly where it looked to make its prediction. 

In this phase, we implemented **Grad-CAM** (Gradient-weighted Class Activation Mapping). This solves the "Black Box" problem of deep learning, ensuring that engineers don't have to blindly trust the AI.

---

## 🔍 1. How Grad-CAM Works (Plain English)

When the CNN (EfficientNet-B0) processes a wafer map, it extracts hundreds of "feature maps"—filters that look for specific patterns like curves, lines, or blobs.

When the model finally predicts "This is an Edge-Ring defect", Grad-CAM asks a simple math question backwards: **"Which of those feature maps voted the strongest for the Edge-Ring conclusion?"**

1. It traces the math backwards from the final prediction to the last image layer (this is called computing the *gradients*).
2. It figures out which pixels in that layer were the most responsible for the final answer.
3. It generates a "Heatmap" where Red/Yellow means "high attention" and Blue means "ignored".
4. We resize that heatmap and overlay it on the original wafer map.

## 🖼️ 2. Visual Results

We ran the trained model from Phase 2 on one random test sample from each of the 9 classes. 

* **Left side:** The original wafer map.
* **Right side:** The Grad-CAM heatmap overlay. Red areas are the "hotspots" where the AI focused its attention to classify the defect.

![Grad-CAM Heatmaps](file:///C:/Users/Peter%20Parker/.gemini/antigravity/brain/f1215d8a-b4d6-4ace-a82c-ce9499d3a549/phase3_gradcam.png)

*(Note: The title color indicates if the model predicted correctly (Green) or incorrectly (Red).)*

### Insights from the Visuals:

* **Center & Donut:** The AI perfectly highlights the central core and the ring pattern. It is looking at exactly the right physical locations.
* **Edge-Loc & Edge-Ring:** The heatmaps strongly activate specifically on the periphery of the wafer, proving the AI learned the concept of "edges".
* **Scratch:** The AI traces the linear path of the scratch. 
* **None (Normal Wafer):** The heatmap is usually diffuse or focused arbitrarily, because there is no specific defect feature to lock onto. Since the model learned that "no defect" equals a clean wafer, diffuse attention is expected.

## ✅ Phase Complete!

By implementing Grad-CAM, we've added a crucial layer of trust. A fab engineer can now look at an AI prediction, see the heatmap highlighting the exact scratch or edge-ring, and instantly verify that the AI isn't hallucinating.

**Next Up — Phase 4:** We will convert these classifications into a business-level metric: a **Yield Risk Score (0-100)** to determine if the manufacturing line needs to be shut down.
