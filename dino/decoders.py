import torch
import torch.nn as nn


class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, patch_h=32, patch_w=32, num_classes=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = patch_w
        self.height = patch_h
        self.classifier = torch.nn.Conv2d(in_channels, num_classes, (1, 1))

    def forward(self, embeddings):
        embeddings = embeddings.reshape(
            -1, self.height, self.width, self.in_channels
        )
        embeddings = embeddings.permute(0, 3, 1, 2)

        return self.classifier(embeddings)
