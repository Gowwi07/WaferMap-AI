"""
src/model.py
============
Phase 2 - EfficientNet-B0 Model Definition

What this file does (plain English):
- Loads EfficientNet-B0 pre-trained on ImageNet (1.2M images)
- Replaces the last layer to output 9 defect classes instead of 1000 ImageNet classes
- Adds Dropout to prevent overfitting
- Provides a summary of the model's architecture
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

from src.dataset import NUM_CLASSES, CLASS_NAMES


def build_model(num_classes=NUM_CLASSES, dropout=0.3, pretrained=True):
    """
    Build and return the EfficientNet-B0 model.

    Args:
        num_classes: number of output classes (9 for our task)
        dropout:     dropout rate — randomly zeroes 30% of activations during
                     training to prevent the model memorizing training data
        pretrained:  if True, loads ImageNet weights (highly recommended)

    Returns:
        model: a PyTorch model ready to train

    ---- HOW TRANSFER LEARNING WORKS ----
    EfficientNet-B0 was trained on ImageNet (1.2M photos of cats, dogs, cars...).
    It learned to detect edges, textures, shapes — universal visual patterns.

    We keep all those learned layers (they detect patterns in wafer maps too).
    We ONLY replace the very last layer (the "classifier head") which originally
    output 1000 ImageNet class scores. Now it outputs 9 wafer defect class scores.

    We "freeze" nothing here — we let all layers train, but with a small learning rate
    for the backbone and a larger one for the new head. This is standard fine-tuning.
    """
    if pretrained:
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model   = models.efficientnet_b0(weights=weights)
        print("[Model] Loaded EfficientNet-B0 with ImageNet pretrained weights.")
    else:
        model = models.efficientnet_b0(weights=None)
        print("[Model] Loaded EfficientNet-B0 without pretrained weights.")

    # ── Replace the classifier head ──────────────────────────────────────────
    # Original classifier: Linear(1280, 1000)  <- 1000 ImageNet classes
    # Our classifier:      Linear(1280, 9)     <- 9 wafer defect classes
    #
    # The 1280 is the "feature vector" size — the compact representation of
    # what the model "saw" in the image. We keep this, only change the output.

    in_features = model.classifier[1].in_features   # = 1280

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    print(f"[Model] Classifier head: Linear({in_features}, {num_classes})")
    print(f"[Model] Dropout rate: {dropout}")

    return model


def count_parameters(model):
    """Count total and trainable parameters in the model."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_summary(model):
    """Print a summary of the model."""
    total, trainable = count_parameters(model)
    print()
    print("=" * 50)
    print("  EfficientNet-B0 Model Summary")
    print("=" * 50)
    print(f"  Total parameters    : {total:>10,}")
    print(f"  Trainable parameters: {trainable:>10,}")
    print(f"  Output classes      : {NUM_CLASSES}")
    print(f"  Class names         : {CLASS_NAMES}")
    print("=" * 50)
    print()


if __name__ == "__main__":
    model = build_model()
    model_summary(model)

    # Test with a dummy input (batch of 2 images, 3 channels, 224x224)
    dummy_input  = torch.randn(2, 3, 224, 224)
    dummy_output = model(dummy_input)
    print(f"  Input shape : {dummy_input.shape}")
    print(f"  Output shape: {dummy_output.shape}")  # Should be [2, 9]
    print("[Test] model.py OK")
