# Phase 1 Evaluation Report – Data Gathering & Exploration

**Status:** COMPLETE  
**Date:** 26 March 2026

---

## What Was Done

| Item | Status |
|---|---|
| Downloaded WM-811K dataset (LSWMD.pkl, ~2GB) | Done |
| Loaded and explored 811,457 wafer records | Done |
| Computed class distribution | Done |
| Generated sample gallery (all 9 defect types) | Done |
| Generated class distribution chart | Done |

---

## Dataset Statistics

```
Total records    :   811,457
Labeled samples  :   811,457
Unlabeled samples:         0

Class Distribution:
  Center      :   4,294  (  0.5%)
  Donut       :     555  (  0.1%)
  Edge-Loc    :   5,189  (  0.6%)
  Edge-Ring   :   9,680  (  1.2%)
  Loc         :   3,593  (  0.4%)
  Near-Full   :     149  (  0.0%)
  Random      :     866  (  0.1%)
  Scratch     :   1,193  (  0.1%)
  None        : 785,938  ( 96.9%)  <-- vast majority!
```

---

## Generated Charts

![Class Distribution](phase1_class_distribution.png)

![Wafer Map Gallery – All 9 Defect Classes](phase1_samples.png)

---

## Key Concept: What Is a Wafer Map?

A **wafer** is a thin, circular disc of silicon — the raw material used to make microchips (like the NAND flash in your SanDisk SSD).

Hundreds of chips are manufactured simultaneously on one wafer. After manufacturing, each chip is tested:
- It either **passes** (good die) or **fails** (bad die)

A **wafer map** is the visual result of this test — a 2D grid where:
- ⬛ Black pixels = background (edge of the wafer, no chip)
- 🔵 Teal pixels = **good die** (chip passed testing)
- 🔴 Red pixels = **defective die** (chip failed)

The *pattern* of red defective dies tells engineers what went wrong in the manufacturing process.

---

## The 9 Defect Classes Explained

| Class | What it looks like | What it means physically |
|---|---|---|
| **None** | All teal, no red | Perfect wafer, no defects |
| **Center** | Red cluster in the middle | Something wrong at the center of the process |
| **Donut** | Red ring around a good center | Edge effects that missed the center |
| **Edge-Loc** | Red patch on one side of the edge | Localized edge contamination |
| **Edge-Ring** | Red ring all around the edge | Uniform edge processing problem |
| **Loc** | Small red cluster anywhere | Local contamination or scratch |
| **Near-Full** | Almost all red, few good dies | Catastrophic process failure |
| **Random** | Scattered red dies everywhere | Random particle contamination |
| **Scratch** | Red line across the wafer | Physical scratch on the surface |

---

## Critical Finding: Class Imbalance

> [!WARNING]
> **96.9% of all wafers are labeled "None" (no defect).** The 8 defect classes together make up only 3.1% of the data.

### Why This Is a Problem

If you trained a model naively, it would learn to always predict "None" and get 96.9% accuracy — but it would **miss every real defect**. That would be useless.

### How We Handle It (Phase 2)

We will use **class weighting** — tell the model to pay extra attention to rare defect classes by giving them a higher penalty when it gets them wrong. This forces the model to learn rare patterns too.

For training, we will **exclude "None" class** (or sample it to match other classes), so the model focuses on learning the actual defect patterns. This is called **class-balanced sampling**.

---

## Dataset Columns

| Column | Description |
|---|---|
| `waferMap` | 2D numpy array — the actual wafer image |
| `dieSize` | Physical size of each die on the wafer |
| `lotName` | Manufacturing lot ID (anonymized) |
| `waferIndex` | Which wafer in the lot (1–25 typically) |
| `trianTestLabel` | 'Training' or 'Test' (pre-split) |
| `failureType` | The defect label (our target to predict) |

---

## What's Next — Phase 2: Model Training

In Phase 2, we will:
1. Build a **data loader** that feeds wafer maps to the model
2. Apply **image preprocessing**: resize to 224×224, normalize pixel values
3. Load **EfficientNet-B0** (pre-trained on ImageNet)
4. Replace its final layer with our 9-class classifier
5. **Train** for several epochs with class balancing
6. Evaluate: accuracy, F1-score, confusion matrix

**Ready to start Phase 2!**
