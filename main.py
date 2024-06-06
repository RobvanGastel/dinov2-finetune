import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.classification import JaccardIndex

import numpy as np
import matplotlib.pyplot as plt

from dino import DINOV2EncoderLoRA, load_voc_dataloader


def visualize_overlay(images, outputs, epoch, batch_idx, n_classes):
    colormap = plt.colormaps["tab20"]
    colors = (
        np.array([colormap(i / n_classes) for i in range(n_classes)])[:, :3]
        * 255
    )

    img = images[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
    output = outputs[0].detach().cpu().numpy()  # (C, H, W)

    output = np.argmax(output, axis=0)  # (H, W)
    img = (img - img.min()) / (img.max() - img.min())
    blended = 0.5 * img + 0.5 * colors[output] / 255.0

    plt.imshow(blended)
    plt.axis("off")

    # Save the figure
    plt.savefig(f"viz_epoch{epoch}_batch{batch_idx}.png")
    plt.close()


def finetune_dino(config, encoder):
    dino_lora = DINOV2EncoderLoRA(
        encoder=encoder,
        r=config.r,
        n=config.n,
        emb_dim=config.emb_dim,
        img_dim=config.img_dim,
        n_classes=config.n_classes,
        use_lora=False,
    ).cuda()

    # Finetuning
    dataloader = load_voc_dataloader(
        img_dim=config.img_dim, batch_size=config.batch_size
    )

    criterion = nn.BCEWithLogitsLoss().cuda()
    iou_metric = JaccardIndex(
        task="multiclass", num_classes=config.n_classes
    ).cuda()
    optimizer = optim.AdamW(dino_lora.parameters(), lr=config.lr)

    for epoch in range(config.epochs):
        dino_lora.train()
        running_loss = 0.0
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.float().cuda()
            masks = masks.float().cuda()

            optimizer.zero_grad()

            logits = dino_lora(images)
            loss = criterion(logits, masks)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Debugging
            if batch_idx % 20 == 0:
                y_hat = F.sigmoid(logits)
                iou_score = iou_metric(y_hat, torch.argmax(masks, dim=1).int())
                print(f"Epoch: {epoch} IoU: {iou_score.item()}")

                visualize_overlay(
                    images, y_hat, epoch, batch_idx, config.n_classes
                )


if __name__ == "__main__":
    # TODO: Argparse config for sizes
    config = argparse.Namespace()
    config.r = 4
    config.size = "large"
    config.batch_size = 12
    config.n_classes = 21

    # Load models with register tokens
    backbones = {
        "small": "vits14_reg",
        "base": "vitb14_reg",
        "large": "vitl14_reg",
        "giant": "vitg14_reg",
    }
    intermediate_layers = {
        "small": [2, 5, 8, 11],
        "base": [2, 5, 8, 11],
        "large": [4, 11, 17, 23],
        "giant": [9, 19, 29, 39],
    }
    embedding_dims = {
        "small": 384,
        "base": 768,
        "large": 1024,
        "giant": 1536,
    }
    config.n = intermediate_layers[config.size]
    config.emb_dim = embedding_dims[config.size]
    config.img_dim = (490, 490)

    # Decoder
    config.epochs = 20
    config.lr = 3e-4
    ###

    encoder = torch.hub.load(
        repo_or_dir="facebookresearch/dinov2",
        model=f"dinov2_{backbones[config.size]}",
    ).cuda()

    finetune_dino(config, encoder)
