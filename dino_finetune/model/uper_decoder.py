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


class PPM(nn.Module):
    """Pyramid Pooling Module in PSPNet"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 128,
        scales: tuple = (1, 2, 3, 6),
    ):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    ConvBNReLU(in_channels, out_channels, 1),
                )
                for scale in scales
            ]
        )

        self.bottleneck = ConvBNReLU(
            in_channels + out_channels * len(scales), out_channels, 3, 1, 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for stage in self.stages:
            outputs.append(
                F.interpolate(
                    stage(x),
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
            )

        outputs = [x] + outputs[::-1]
        return self.bottleneck(torch.cat(outputs, dim=1))


class UperDecoder(nn.Module):
    """The UperHead decoder head is used in most papers when they mention using a decoder head for
    image segmentation.

    Xiao, T., Liu, Y., Zhou, B., Jiang, Y., & Sun, J. (2018). Unified Perceptual Parsing for Scene
    Understanding. Computer Vision - ECCV 2018 (Vol. 11209, pp. 432-448).
    https://arxiv.org/abs/1807.10221
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scales: tuple = (1, 2, 3, 6),
        patch_h: int = 35,
        patch_w: int = 35,
        n_classes: int = 100,
    ):
        super().__init__()
        self.width = patch_w
        self.height = patch_h
        self.in_channels = in_channels
        self.out_channels = out_channels

        # PPM Module
        self.ppm = PPM(in_channels[-1], out_channels, scales=scales)

        # FPN Module
        self.fpn_in = nn.ModuleList()
        self.fpn_out = nn.ModuleList()

        for i in in_channels[:-1]:
            self.fpn_in.append(ConvBNReLU(i, out_channels, 1))
            self.fpn_out.append(ConvBNReLU(out_channels, out_channels, 3, 1, 1))

        self.bottleneck = ConvBNReLU(
            len(in_channels) * out_channels, out_channels, 3, 1, 1
        )
        self.dropout = nn.Dropout2d(0.1)
        self.conv = nn.Conv2d(out_channels, n_classes, 1)

    def forward(self, features: tuple) -> torch.Tensor:
        f = self.ppm(features[-1])
        fpn_features = [f]

        for i in reversed(range(len(features) - 1)):
            feature = self.fpn_in[i](features[i])
            f = feature + F.interpolate(
                f, size=feature.shape[-2:], mode="bilinear", align_corners=False
            )
            fpn_features.append(self.fpn_out[i](f))

        fpn_features.reverse()
        for i in range(1, len(features)):
            fpn_features[i] = F.interpolate(
                fpn_features[i],
                size=fpn_features[0].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        output = self.bottleneck(torch.cat(fpn_features, dim=1))
        output = self.conv(self.dropout(output))
        return output
