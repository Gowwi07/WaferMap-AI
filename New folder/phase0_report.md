# Phase 0 Evaluation Report – Environment Setup

**Status:** COMPLETE  
**Date:** 26 March 2026  
**Python:** 3.14.3 | **PyTorch:** 2.11.0+cpu | **Mode:** CPU (no GPU detected)

---

## What Was Done

| Item | Status |
|---|---|
| Project folders created (`data/`, `models/`, `src/`, `notebooks/`, `reports/`) | Done |
| [requirements.txt](file:///e:/Projects/SanDisk/requirements.txt) created | Done |
| [setup_check.py](file:///e:/Projects/SanDisk/setup_check.py) — verification script | Done |
| [README.md](file:///e:/Projects/SanDisk/README.md) — project overview | Done |
| All 12 Python packages installed | Done |

---

## Verification Result

All 12/12 packages passed:

```
[OK]  PyTorch         - Deep learning framework
[OK]  TorchVision     - Image datasets & transforms
[OK]  NumPy           - Numerical computation
[OK]  Pandas          - Data analysis
[OK]  Scikit-learn    - ML metrics & utilities
[OK]  Matplotlib      - Plotting
[OK]  Seaborn         - Statistical plots
[OK]  OpenCV          - Image processing
[OK]  Pillow          - Image I/O
[OK]  Streamlit       - Web dashboard
[OK]  TQDM            - Progress bars
[OK]  SciPy           - Scientific computation

[CPU] No GPU - will run on CPU
ALL 12/12 CHECKS PASSED -- Ready for Phase 1!
```

---

## What Each Library Does (Beginner Explanation)

### PyTorch
This is the **brain** of the project. PyTorch is a deep learning library — it lets you define neural networks (AI models) and train them on data. Think of it as the engine of a car: the most important part that makes everything else work.

### TorchVision
An extension of PyTorch that provides:
- Pre-built datasets (like ImageNet)
- Pre-trained models (like EfficientNet-B0 that we'll use)
- Image transformation tools (resizing, normalizing, flipping images for training)

### NumPy
For numerical computations. When you have a 2D grid of numbers (like a wafer map image), NumPy makes it easy to do math on it. Think of it as a very powerful spreadsheet.

### Pandas
For loading and analyzing data tables. The WM-811K dataset comes as a `.csv` or `.pkl` file — Pandas reads it and lets you filter and inspect it easily.

### Scikit-learn
Provides tools for evaluating ML models:
- `accuracy_score` — how many predictions were correct
- `f1_score` — a better metric when class sizes are unequal
- `confusion_matrix` — shows which classes get confused with each other

### Matplotlib & Seaborn
For creating graphs and charts. We'll use these to plot training curves, confusion matrices, and defect sample galleries.

### OpenCV (`cv2`)
For image processing — reading images, resizing them, overlaying colored heatmaps on top of wafer maps (used in Phase 3).

### Pillow (PIL)
Python's main library for loading and saving image files (PNG, JPG, etc.).

### Streamlit
Creates a web interface from pure Python — no HTML or CSS needed. We'll use it in Phase 5 to build the upload-and-predict dashboard.

### TQDM
Adds a progress bar to loops. When training takes 30 minutes, you want to see "Epoch 5/20: 234/500 batches [=====>...]" instead of staring at a blank screen.

### SciPy
Scientific computing tools. We'll use it for some image processing utilities.

---

## CPU vs GPU — What It Means For You

> [!NOTE]
> Your computer has no dedicated GPU (graphics card). This means training will run on the CPU, which is slower, but the results are **identical**.

| | CPU | GPU |
|---|---|---|
| Training speed | ~30–60 min | ~2–5 min |
| Results | Same accuracy | Same accuracy |
| Memory | System RAM | GPU VRAM |

**For this project:** We'll use a smaller version of the dataset or fewer epochs to keep CPU training time reasonable. This is perfectly fine for the hackathon.

---

## Key Concept: What Is a Neural Network?

A neural network is a math function that learns from examples. You show it thousands of wafer map images with known labels ("this one has an Edge-Ring defect"), and it learns to recognize the patterns.

It does this in **layers**:
1. First layers detect simple patterns (edges, curves)
2. Middle layers detect complex shapes ("ring-like pattern in this region")
3. Last layer decides: "this is most likely an Edge-Ring defect (87% confidence)"

---

## Key Concept: What Is Transfer Learning?

Training a neural network from scratch requires millions of images. We don't have that. Instead, we use **transfer learning**: start with EfficientNet-B0, which was already trained on 1.2 million general images (ImageNet). It already knows how to detect edges, textures, and shapes.

We then fine-tune it on our 811K wafer maps. It adapts its existing knowledge to our specific task — much faster and more accurate than starting from zero.

---

## What's Next — Phase 1: Data Gathering & Exploration

In Phase 1, we will:
1. Download the WM-811K dataset (a famous benchmark of 811,457 labeled wafer maps)
2. Explore how many samples exist per defect class
3. Visualize sample wafer maps for each of the 9 defect types
4. Understand why "class imbalance" is a challenge and how we handle it

**Ready to start Phase 1 when you are!**
