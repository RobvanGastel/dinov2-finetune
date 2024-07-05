import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        pad: int = 0,
    ):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super().__init__(*layers)


class FPNDecoder(nn.Module):
    """The Feature Pyramid Network (FPN) decoder is used as a module for
    decoders in image segmentation to do multiscale spatial pooling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        inter_layers: int,
        patch_h: int = 35,
        patch_w: int = 35,
        n_classes: int = 100,
    ):
        super().__init__()
        self.width = patch_w
        self.height = patch_h
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_layers = inter_layers

        inner_channels = [
            out_channels,
            out_channels // 2,
            out_channels // 4,
            out_channels // 6,
        ]

        self.upsample = nn.Upsample(scale_factor=2)

        # FPN Module
        self.conv1 = ConvBNReLU(in_channels, out_channels, 3)
        self.conv2 = ConvBNReLU(out_channels, inner_channels[1], 3)
        self.conv3 = ConvBNReLU(inner_channels[1], inner_channels[2], 3)
        self.conv4 = ConvBNReLU(inner_channels[2], inner_channels[3], 3)
        self.conv5 = ConvBNReLU(inner_channels[3], n_classes, 3)

        # Intermediate layers
        self.inter_conv1 = nn.Conv2d(in_channels, inner_channels[1], 1)
        self.inter_conv2 = nn.Conv2d(in_channels, inner_channels[2], 1)
        self.inter_conv3 = nn.Conv2d(in_channels, inner_channels[3], 1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        x = features[-1]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)

        inter_fpn = self.inter_conv1(features[0])
        x = x + F.interpolate(inter_fpn, size=x.shape[-2:], mode="nearest")
        x = self.conv3(x)
        x = self.upsample(x)

        inter_fpn = self.inter_conv2(features[1])
        x = x + F.interpolate(inter_fpn, size=x.shape[-2:], mode="nearest")
        x = self.conv4(x)
        x = self.upsample(x)

        inter_fpn = self.inter_conv3(features[2])
        x = x + F.interpolate(inter_fpn, size=x.shape[-2:], mode="nearest")
        return self.conv5(x)
