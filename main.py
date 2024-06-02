import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from dino import DINOV2EncoderLoRA, load_seg_dataloader


def finetune_dino(config, encoder):
    # TODO: Load head

    dino_lora = DINOV2EncoderLoRA(
        encoder=encoder,
        decoder=None,
        r=config.r,
        n=config.n,
        emb_dim=config.emb_dim,
        img_dim=config.img_dim,
    )

    # Finetuning
    # TODO: Finetune
    dataloader = load_seg_dataloader(img_size=config.img_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(dino_lora.parameters(), lr=config.lr)

    for epoch in range(config.epochs):
        dino_lora.train()
        running_loss = 0.0
        for images, masks in dataloader:
            optimizer.zero_grad()

            outputs = dino_lora(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()


if __name__ == "__main__":
    # TODO: Argparse config for sizes
    config = argparse.Namespace()
    config.r = 4
    config.size = "large"

    # TODO: With register _reg_lc
    backbones = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
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
    config.img_dim = (520, 520)

    # Decoder

    config.epochs = 200
    config.lr = 3e-4
    ###

    encoder = torch.hub.load(
        repo_or_dir="facebookresearch/dinov2",
        model=f"dinov2_{backbones[config.size]}",
    ).cuda()

    finetune_dino(config, encoder)
