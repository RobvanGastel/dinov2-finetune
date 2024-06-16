import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(
        self,
        channels: int,
        patch_h: int = 35,
        patch_w: int = 35,
        n_classes: int = 1000,
    ):
        super().__init__()
        self.channels = channels
        self.width = patch_w
        self.height = patch_h
        self.classifier = nn.Conv2d(channels, n_classes, (1, 1))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = embeddings.reshape(-1, self.height, self.width, self.channels)
        embeddings = embeddings.permute(0, 3, 1, 2)
        return self.classifier(embeddings)
