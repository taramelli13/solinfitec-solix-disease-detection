"""Grad-CAM visualization for Swin Transformer."""

import logging
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger("solinfitec.gradcam")


def swin_reshape_transform(tensor: torch.Tensor, height: int = 7, width: int = 7):
    """Reshape Swin Transformer output for Grad-CAM.

    Swin outputs (B, H*W, C) -> we need (B, C, H, W).
    """
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result


class SwinGradCAM:
    """Grad-CAM for Swin Transformer (Stage 3 target).

    Args:
        model: SwinClassifier model.
        target_layer: Name/index of the target layer (default: last stage).
    """

    def __init__(self, model, target_layer: str = "stage3"):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Register hooks on the target layer
        target = self._get_target_layer(target_layer)
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def _get_target_layer(self, name: str):
        """Get target layer from backbone."""
        if name == "stage3" and hasattr(self.model.backbone, "layers"):
            return self.model.backbone.layers[-1]
        elif name == "stage2" and hasattr(self.model.backbone, "layers"):
            return self.model.backbone.layers[-2]
        # Fallback: use norm layer
        if hasattr(self.model.backbone, "norm"):
            return self.model.backbone.norm
        raise ValueError(f"Cannot find target layer: {name}")

    def _save_activation(self, module, input, output):
        if isinstance(output, torch.Tensor):
            self.activations = output.detach()
        elif isinstance(output, tuple):
            self.activations = output[0].detach()

    def _save_gradient(self, module, grad_input, grad_output):
        if isinstance(grad_output, tuple):
            self.gradients = grad_output[0].detach()
        else:
            self.gradients = grad_output.detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap.

        Args:
            input_tensor: (1, 3, H, W) input image tensor.
            target_class: Target class index. If None, uses predicted class.

        Returns:
            heatmap: (H, W) numpy array normalized to [0, 1].
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None or self.activations is None:
            logger.warning("No gradients/activations captured")
            return np.zeros((7, 7))

        gradients = self.gradients
        activations = self.activations

        # Handle Swin's (B, N, C) format
        if gradients.dim() == 3:
            # (B, N, C) -> pool over spatial, weight channels
            weights = gradients.mean(dim=1, keepdim=True)  # (B, 1, C)
            cam = (weights * activations).sum(dim=-1)  # (B, N)
            h = w = int(cam.size(1) ** 0.5)
            cam = cam.reshape(1, h, w)
        else:
            # (B, C, H, W) standard CNN format
            weights = gradients.mean(dim=[2, 3], keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = cam.squeeze(1)

        # ReLU + normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        heatmap = cam.squeeze().cpu().numpy()

        # Resize to input image size
        h_in, w_in = input_tensor.shape[2], input_tensor.shape[3]
        heatmap = cv2.resize(heatmap, (w_in, h_in))

        return heatmap

    def visualize(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        save_path: Optional[str] = None,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Overlay Grad-CAM heatmap on original image.

        Args:
            image: (H, W, 3) RGB image in [0, 255].
            heatmap: (H, W) heatmap in [0, 1].
            save_path: Optional save path.
            alpha: Overlay transparency.

        Returns:
            overlay: (H, W, 3) image with heatmap overlay.
        """
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        overlay = (alpha * heatmap_colored + (1 - alpha) * image).astype(np.uint8)

        if save_path:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(image)
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(heatmap, cmap="jet")
            axes[1].set_title("Grad-CAM Heatmap")
            axes[1].axis("off")

            axes[2].imshow(overlay)
            axes[2].set_title("Overlay")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Grad-CAM saved to {save_path}")

        return overlay
