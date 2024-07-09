import json
import logging
import argparse
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.classification import JaccardIndex as IoU

from dino_finetune import DINOV2EncoderLoRA, get_dataloader, visualize_overlay


def validate_epoch(
    dino_lora: nn.Module,
    val_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    f_iou: Callable,
    metrics: dict,
) -> None:
    val_loss = 0.0
    val_iou = 0.0

    dino_lora.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.float().cuda()
            masks = masks.long().cuda()

            logits = dino_lora(images)
            loss = criterion(logits, masks)
            val_loss += loss.item()

            y_hat = torch.sigmoid(logits)
            iou_score = f_iou(y_hat, masks)
            val_iou += iou_score.item()

    metrics["val_loss"].append(val_loss / len(val_loader))
    metrics["val_iou"].append(val_iou / len(val_loader))


def finetune_dino(config: argparse.Namespace, encoder: nn.Module):
    dino_lora = DINOV2EncoderLoRA(
        encoder=encoder,
        r=config.r,
        emb_dim=config.emb_dim,
        img_dim=config.img_dim,
        n_classes=config.n_classes,
        use_lora=config.use_lora,
        use_fpn=config.use_fpn,
    ).cuda()

    if config.lora_weights:
        dino_lora.load_parameters(config.lora_weights)

    train_loader, val_loader = get_dataloader(
        config.dataset, img_dim=config.img_dim, batch_size=config.batch_size
    )

    # Finetuning for segmentation
    criterion = nn.CrossEntropyLoss(ignore_index=255).cuda()
    f_iou = IoU(
        task="multiclass",
        num_classes=config.n_classes,
        ignore_index=255,
        average="micro",
    ).cuda()
    optimizer = optim.AdamW(dino_lora.parameters(), lr=config.lr)

    # Log training and validation metrics
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
    }

    for epoch in range(config.epochs):
        dino_lora.train()

        for images, masks in train_loader:
            images = images.float().cuda()
            masks = masks.long().cuda()
            optimizer.zero_grad()

            logits = dino_lora(images)
            loss = criterion(logits, masks)

            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            y_hat = torch.sigmoid(logits)
            validate_epoch(dino_lora, val_loader, criterion, f_iou, metrics)
            dino_lora.save_parameters(f"output/{config.exp_name}.pt")

            if config.debug:
                # Visualize some of the batch and write to files when debugging
                visualize_overlay(
                    images, y_hat, config.n_classes, filename=f"viz_{epoch}"
                )

            logging.info(
                f"Epoch: {epoch} - val IoU: {metrics['val_iou'][-1]} "
                f"- val loss {metrics['val_loss'][-1]}"
            )

    # Log metrics & save model the final values
    # Saves only loRA parameters and classifer
    dino_lora.save_parameters(f"output/{config.exp_name}.pt")
    with open(f"output/{config.exp_name}_metrics.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="lora",
        help="Experiment name",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug by visualizing some of the outputs to file for a sanity check",
    )
    parser.add_argument(
        "--r",
        type=int,
        default=3,
        help="loRA rank parameter r",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="large",
        help="DINOv2 backbone parameter [small, base, large, giant]",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use Low-Rank Adaptation (LoRA) to finetune",
    )
    parser.add_argument(
        "--use_fpn",
        action="store_true",
        help="Use the FPN decoder for finetuning",
    )
    parser.add_argument(
        "--img_dim",
        type=int,
        nargs=2,
        default=(490, 490),
        help="Image dimensions (height width)",
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        default=None,
        help="Load the LoRA weights from file location",
    )

    # Training parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="ade20k",
        help="The dataset to finetune on, either `voc` or `ade20k`",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=12,
        help="Finetuning batch size",
    )
    config = parser.parse_args()

    # All backbone sizes and configurations
    backbones = {
        "small": "vits14_reg",
        "base": "vitb14_reg",
        "large": "vitl14_reg",
        "giant": "vitg14_reg",
    }
    embedding_dims = {
        "small": 384,
        "base": 768,
        "large": 1024,
        "giant": 1536,
    }
    config.emb_dim = embedding_dims[config.size]

    # Dataset
    dataset_classes = {
        "voc": 21,
        "ade20k": 150,
    }
    config.n_classes = dataset_classes[config.dataset]

    encoder = torch.hub.load(
        repo_or_dir="facebookresearch/dinov2",
        model=f"dinov2_{backbones[config.size]}",
    ).cuda()

    finetune_dino(config, encoder)
