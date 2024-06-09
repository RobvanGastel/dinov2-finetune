import torch
import torch.nn as nn


class InstanceClassifier(nn.Module):
    def __init__(self, channels, patch_h=35, patch_w=35, n_classes=1000):
        super().__init__()
        self.channels = channels
        self.width = patch_w
        self.height = patch_h

        self.semantic_head = nn.Conv2d(channels, n_classes, (1, 1))
        self.center_head = nn.Conv2d(channels, 1, (1, 1))
        self.offset_head = nn.Conv2d(channels, 2, (1, 1))

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.channels)
        embeddings = embeddings.permute(0, 3, 1, 2)

        # Semantic
        semantic = self.semantic_head(embeddings)

        # Instance
        instance_centers = self.center_head(embeddings)
        instance_offsets = self.offset_head(embeddings)

        return semantic, instance_centers, instance_offsets
