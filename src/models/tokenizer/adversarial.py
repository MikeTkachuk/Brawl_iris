import torch
from torch import nn
from torchvision.transforms.functional import gaussian_blur


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None, conv_shortcut: bool = False,
                 dropout=0.0) -> None:
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.norm2 = nn.BatchNorm2d(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.nonlinearity = nn.Mish()
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AdversarialLoss(nn.Module):
    def __init__(self, channels=(16, 16, 32, 64, 128),
                 resnet_block=(False, True, True, True, False)):
        super().__init__()
        self.backbone = nn.ModuleList()
        channels = (3,) + channels
        for depth, is_resnet in zip(range(len(channels) - 1), resnet_block):
            if is_resnet:
                block = nn.Sequential(
                    ResnetBlock(channels[depth], channels[depth + 1]),
                    nn.AvgPool2d(kernel_size=2, stride=2)
                )
            else:
                block = nn.Sequential(
                    nn.Conv2d(channels[depth], channels[depth + 1], kernel_size=3,
                              padding=1),
                    nn.BatchNorm2d(channels[depth + 1]),
                    nn.Mish(),
                    nn.AvgPool2d(kernel_size=2, stride=2)
                )
            self.backbone.append(block)
        self.discriminator = nn.Sequential(
            nn.LazyLinear(64),
            nn.BatchNorm1d(64),
            nn.Mish(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.Mish(),
            nn.Linear(16, 1)
        )

    def forward(self, x, blur=True):
        if blur:
            x = gaussian_blur(x, kernel_size=5)
        intermediate_features = []
        for block in self.backbone:
            x = block(x)
            intermediate_features.append(x.mean((-1, -2)))
        features = torch.cat(intermediate_features, dim=-1)
        return self.discriminator(features)