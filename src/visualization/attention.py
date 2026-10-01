import os
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import cv2
from typing import List, Union, Optional

def visualize_attention_heatmap(
    image: Union[str, Image.Image],
    tokens: List[str],
    alphas: List[torch.Tensor],
    save_path: Optional[str] = None,
    smooth: bool = True
) -> plt.Figure:
    """
    Generate visual attention heatmap overlays for each generated word token.
    Args:
        image: File path or PIL Image object
        tokens: List of word token strings
        alphas: List of 1D tensors (num_pixels,) or 2D tensors (7, 7) attention weights
        save_path: Output file path to save the generated figure
        smooth: Apply gaussian blur smoothing to attention overlay
    Returns:
        matplotlib Figure object
    """
    if isinstance(image, str):
        pil_img = Image.open(image).convert("RGB")
    else:
        pil_img = image.convert("RGB")

    img_np = np.array(pil_img)
    orig_h, orig_w = img_np.shape[:2]

    num_words = min(len(tokens), len(alphas))
    if num_words == 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(pil_img)
        ax.axis("off")
        return fig

    # Calculate grid layout for subplots
    num_cols = min(4, num_words + 1)
    num_rows = math.ceil((num_words + 1) / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3.5, num_rows * 3.5))
    if num_rows == 1 and num_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # 1. First plot: Original Image
    axes[0].imshow(pil_img)
    axes[0].set_title("Original Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. Subsequent plots: Word + Attention Heatmap
    for i in range(num_words):
        ax = axes[i + 1]
        token = tokens[i]
        alpha = alphas[i]

        if isinstance(alpha, torch.Tensor):
            alpha_np = alpha.detach().cpu().numpy()
        else:
            alpha_np = np.array(alpha)

        # Reshape to 2D spatial grid (e.g., 7x7 or 14x14)
        grid_dim = int(np.sqrt(alpha_np.shape[0]))
        alpha_grid = alpha_np.reshape(grid_dim, grid_dim)

        # Resize heatmap to match original image dimensions
        heatmap = cv2.resize(alpha_grid, (orig_w, orig_h))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        if smooth:
            heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)

        # Plot base image + heatmap overlay
        ax.imshow(pil_img)
        ax.imshow(heatmap, cmap="jet", alpha=0.55)
        ax.set_title(f"Word: '{token}'", fontsize=11, fontweight="bold")
        ax.axis("off")

    # Turn off extra unused axes
    for j in range(num_words + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig
