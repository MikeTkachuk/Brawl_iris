import torch
from torch import nn
from torchvision.transforms.functional import gaussian_blur


class AdversarialLoss(nn.Module):
    def __init__(self, channels=(16, 32, 64, 128)):
        super().__init__()
        self.backbone = nn.ModuleList()
        channels = (3,) + channels

        for depth in range(len(channels) - 1):
            block = nn.Sequential(
                nn.Conv2d(channels[depth], channels[depth + 1], kernel_size=3,
                          padding=1),
                nn.BatchNorm2d(channels[depth + 1]),
                nn.Mish(),
                nn.AvgPool2d(kernel_size=2, stride=2)
            )
            self.backbone.append(block)
        self.discriminator = nn.LazyLinear(1)

    def forward(self, x, blur=True):
        if blur:
            x = gaussian_blur(x, kernel_size=3)
        intermediate_features = []
        for block in self.backbone:
            x = block(x)
            intermediate_features.append(x.mean((-1, -2)))
        features = torch.cat(intermediate_features, dim=-1)
        return self.discriminator(features)