from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize_overlay(
    images: torch.Tensor,
    masks: torch.Tensor,
    n_classes: int,
    filename: Optional[str | None] = None,
) -> Optional[None | np.ndarray]:
    """Overlaying the images with the mask labels

    Args:
        images (torch.Tensor): The batch of images of which the first is taken.
        masks (torch.Tensor): The batch of labels of which the first is taken.
        n_classes (int): The number of classes expected in the channel C dimension of
            (B, C, H, W)
        filename (Optional[str  |  None], optional): When a filename is provided the
            file is saved to the root directory of the project for debugging. Defaults to None.

    Returns:
        np.ndarray: If no filename is specified.
    """
    colormap = plt.colormaps["tab20"]
    colors = np.array([colormap(i / n_classes) for i in range(n_classes)])[:, :3] * 255

    img = images[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
    mask = masks[0].detach().cpu().numpy()  # (C, H, W)

    mask = np.argmax(mask, axis=0)  # (H, W)
    img = (img - img.min()) / (img.max() - img.min())
    overlayed_img = 0.5 * img + 0.5 * colors[mask] / 255.0

    if filename:
        plt.imshow(overlayed_img)
        plt.axis("off")
        plt.savefig(f"{filename}.png")
        plt.close()
    else:
        return overlayed_img
