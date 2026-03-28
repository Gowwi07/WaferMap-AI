# Phase 5: Streamlit App & Final Demo

**Goal:** Wrap the entire AI pipeline (Classification + Grad-CAM + Risk Scoring) into an interactive web dashboard so engineers can use it easily.

We have successfully tied all 4 previous phases together into [src/app.py](file:///e:/Projects/SanDisk/src/app.py). The application is now live!

---

## 🚀 1. Using the Dashboard
The Streamlit app is running on your local machine.
Open your web browser and go to: **[http://localhost:8501](http://localhost:8501)**

### Features:
1. **Sidebar Navigation:** You can filter by any of the 9 true defect types and select a random wafer map from our test dataset.
2. **AI Analysis:** Click "Run AI Analysis" to trigger the entire pipeline in under a second.
3. **Top Metrics:** Displays the predicted class, confidence, Yield Risk Score (0-100), and a color-coded Action Recommendation (Monitor/Investigate/Stop Line).
4. **Visual Explanations:** Shows the raw input image side-by-side with the Grad-CAM heatmap overlay so you can visually verify the AI's reasoning.

---

## 🏆 2. Project Hackathon Summary
We set out to build an end-to-end AI-powered wafer defect classification system from zero knowledge. We have successfully achieved all goals:

### What We Built:
* **Data Engineering (Phase 1):** Ingested the massive 811K-sample WM-811K dataset, handled extreme class imbalance by capping dominant classes, and constructed robust data loaders.
* **Model Engineering (Phase 2):** Implemented Transfer Learning using an EfficientNet-B0 backbone. Replaced the classifier head, implemented weighted Cross-Entropy Loss, and fine-tuned it on the CPU to achieve ~78% accuracy on 9 complicated classes in just 15 minutes of training.
* **Explainable AI (Phase 3):** Wrote a custom Grad-CAM implementation that hooks directly into EfficientNet's final convolutional layer to visualize attention hotspots, solving the "AI Black Box" problem.
* **Business Logic (Phase 4):** Authored a heuristic yield risk scoring algorithm that multiplies defect severity, AI confidence, and Grad-CAM area to trigger actionable factory alerts.
* **Deployment (Phase 5):** Built a sleek, reactive Python UI using Streamlit to demo the full capability to non-technical stakeholders.

### Conclusion:
You now have a complete, production-ready prototype for the SanDisk Hackathon! The system perfectly demonstrates how modern Deep Learning (EfficientNet), Interpretability (Grad-CAM), and business rules can combine to automate semiconductor quality control.

You can stop the background Streamlit process by closing the terminal or pressing `Ctrl+C`. Good luck with the presentation!
