"""
src/gradcam.py
==============
Phase 3 - Grad-CAM Implementation

Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique for making
CNN-based models more interpretable. It tells us *where* the model is looking
to make its prediction.

How it works (plain English):
1. We feed an image forward through the model.
2. We see what the model predicted (e.g., "Donut" defect).
3. We calculate the *gradients* (how much each pixel in the final convolutional layer contributed to that "Donut" prediction).
4. We combine the layer's activations and the gradients to create a heatmap.
5. The "hot" spots on the map show the exact pixels the AI used to guess "Donut".
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Args:
            model: The trained neural network (EfficientNet-B0)
            target_layer: The last convolutional layer in the network.
        """
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None

        # These "hooks" attach to the target layer and steal its outputs 
        # and gradients during the forward/backward passes.
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """Steal the activations (feature maps) from the forward pass."""
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        """Steal the gradients from the backward pass."""
        # grad_output represents gradients flowing backwards
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        """
        Generate the Grad-CAM heatmap.
        Args:
            x: Input image tensor (1, C, H, W)
            class_idx: The class we want an explanation for. If None, uses the model's top prediction.
        """
        self.model.eval()

        # 1. Forward Pass to get prediction
        output = self.model(x)

        if class_idx is None:
            # If no class specified, use the highest scoring class
            class_idx = output.argmax(dim=1).item()

        # 2. Backward Pass to get gradients for that specific class
        self.model.zero_grad()
        # We only want the gradient for the chosen class, so we create a dummy "1" for it
        class_loss = output[0, class_idx]
        class_loss.backward(retain_graph=True)

        # 3. Combine gradients and activations
        # We take the global average of the gradients
        gradients = self.gradients.cpu().data.numpy()[0]   # Shape: (Channels, H, W)
        activations = self.activations.cpu().data.numpy()[0] # Shape: (Channels, H, W)

        # Average across the spatial dimensions (H, W) to get a single weight per channel
        weights = np.mean(gradients, axis=(1, 2))

        # Multiply each activation channel by its weight
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # 4. Post-processing (ReLU and Normalize)
        # We only care about features that positively affect the class (ReLU)
        cam = np.maximum(cam, 0)
        
        # Resize to input layer size
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        
        # Normalize between 0 and 1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, output, class_idx

def overlay_heatmap(img_rgb, heatmap_map):
    """
    Overlays a Grad-CAM heatmap on top of the original image limit.
    Args:
        img_rgb: Original image (H, W, 3) normalized to 0-1 or 0-255
        heatmap_map: 2D array (H, W) normalized to 0-1
    """
    if img_rgb.max() <= 1.0:
        img_rgb = (img_rgb * 255).astype(np.uint8)

    # Convert heatmap to false color (JET colormap)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_map), cv2.COLORMAP_JET)

    # Blend original image and heatmap (0.6 opacity original, 0.4 opacity heatmap)
    overlaid = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)
    return overlaid
