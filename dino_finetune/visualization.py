import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize_overlay(
    images: torch.Tensor,
    masks: torch.Tensor,
    n_classes: int,
    filename="viz",
) -> None:
    colormap = plt.colormaps["tab20"]
    colors = np.array([colormap(i / n_classes) for i in range(n_classes)])[:, :3] * 255

    img = images[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
    mask = masks[0].detach().cpu().numpy()  # (C, H, W)

    mask = np.argmax(mask, axis=0)  # (H, W)
    img = (img - img.min()) / (img.max() - img.min())

    plt.imshow(0.5 * img + 0.5 * colors[mask] / 255.0)
    plt.axis("off")
    plt.savefig(f"{filename}.png")
    plt.close()
