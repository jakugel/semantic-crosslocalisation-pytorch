import torch.nn as nn
from utility_layers import EqualizedConv2d, EqualizedLinear, EqualizedMinibatchDiscrimination
from math import sqrt
import torch


class DiscriminatorImage(nn.Module):
    def __init__(self, log_resolution, n_features=64, max_features=256, num_layers=4,
                 mdl_num_kernels=25, mdl_kernel_size=15, equalized=True, dropout=None):
        super().__init__()

        features = [min(max_features, n_features * (2 ** i)) for i in range(num_layers - 1)]

        if equalized:
            conv2d_l = EqualizedConv2d(1, n_features, 1)
        else:
            conv2d_l = nn.Conv2d(1, n_features, 1)

        self.from_rgb = nn.Sequential(
            conv2d_l,
            nn.LeakyReLU(0.2, True),
        )
        n_blocks = len(features) - 1
        blocks = [DiscriminatorBlock(features[i], features[i + 1], equalized, dropout) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)

        final_channels = features[-1]  # Number of channels after last conv block
        final_resolution = 2 ** (log_resolution - (n_blocks))  # Spatial size after downsampling
        flattened_features = final_channels * (final_resolution ** 2)

        self.mdl1 = EqualizedMinibatchDiscrimination(flattened_features, mdl_num_kernels, mdl_kernel_size, equalized=equalized)
        self.mdl2 = EqualizedMinibatchDiscrimination(flattened_features, mdl_num_kernels, mdl_kernel_size, equalized=equalized)

        final_features = flattened_features + mdl_num_kernels

        # two mdl heads, one for labelled and one for unlabelled images,
        # given their batch distributions are likely to differ
        if equalized:
            self.final1 = EqualizedLinear(final_features, 1)
            self.final2 = EqualizedLinear(final_features, 1)
        else:
            self.final1 = nn.Linear(final_features, 1)
            self.final2 = nn.Linear(final_features, 1)

    def forward(self, x):
        x = self.from_rgb(x)
        x = self.blocks(x)

        x = nn.Flatten()(x)
        x1 = self.mdl1(x)
        x1 = x1.reshape(x1.shape[0], -1)

        x2 = self.mdl2(x)
        x2 = x2.reshape(x2.shape[0], -1)

        return self.final1(x1), self.final2(x2)


class DiscriminatorMask(nn.Module):
    def __init__(self, log_resolution, n_features=64, max_features=256, num_layers=4, mdl_num_kernels=25, mdl_kernel_size=15,
                 equalized=True, dropout=None):
        super().__init__()

        features = [min(max_features, n_features * (2 ** i)) for i in range(num_layers - 1)]

        if equalized:
            conv2d_l = EqualizedConv2d(1, n_features, 1)
        else:
            conv2d_l = nn.Conv2d(1, n_features, 1)

        self.from_rgb = nn.Sequential(
            conv2d_l,
            nn.LeakyReLU(0.2, True),
        )

        n_blocks = len(features) - 1
        blocks = [DiscriminatorBlock(features[i], features[i + 1], equalized, dropout) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)

        final_channels = features[-1]  # Number of channels after last conv block
        final_resolution = 2 ** (log_resolution - (n_blocks))  # Spatial size after downsampling
        flattened_features = final_channels * (final_resolution ** 2)

        self.mdl = EqualizedMinibatchDiscrimination(flattened_features, mdl_num_kernels, mdl_kernel_size, equalized=equalized)

        final_features = flattened_features + mdl_num_kernels

        if equalized:
            self.final = EqualizedLinear(final_features, 1)
        else:
            self.final = nn.Linear(final_features, 1)

    def forward(self, x):
        x = self.from_rgb(x)
        x = self.blocks(x)

        x = nn.Flatten()(x)
        x = self.mdl(x)
        x = x.reshape(x.shape[0], -1)

        return self.final(x)


class DiscriminatorJoint(nn.Module):
    def __init__(self, log_resolution, n_features=64, max_features=256, num_layers=4, mdl_num_kernels=25, mdl_kernel_size=15,
                 equalized=True, dropout=None):
        super().__init__()

        features = [min(max_features, n_features * (2 ** i)) for i in range(num_layers - 1)]

        if equalized:
            conv2d_l = EqualizedConv2d(2, n_features, 1)
        else:
            conv2d_l = nn.Conv2d(2, n_features, 1)

        self.from_rgb = nn.Sequential(
            conv2d_l,
            nn.LeakyReLU(0.2, True),
        )

        n_blocks = len(features) - 1
        blocks = [DiscriminatorBlock(features[i], features[i + 1], equalized, dropout) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)

        final_channels = features[-1]  # Number of channels after last conv block
        final_resolution = 2 ** (log_resolution - (n_blocks))  # Spatial size after downsampling
        flattened_features = final_channels * (final_resolution ** 2)

        self.mdl = EqualizedMinibatchDiscrimination(flattened_features, mdl_num_kernels, mdl_kernel_size, equalized=equalized)

        final_features = flattened_features + mdl_num_kernels

        if equalized:
            self.final = EqualizedLinear(final_features, 1)
        else:
            self.final = nn.Linear(final_features, 1)

    def forward(self, x, m):
        x = torch.cat((x, m), axis=1)
        x = self.from_rgb(x)
        x = self.blocks(x)

        x = nn.Flatten()(x)
        x = self.mdl(x)
        x = x.reshape(x.shape[0], -1)

        return self.final(x)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_features, out_features, equalized=True, dropout=None):
        super().__init__()
        if equalized:
            conv2d_r = EqualizedConv2d(in_features, out_features, kernel_size=1)
        else:
            conv2d_r = nn.Conv2d(in_features, out_features, kernel_size=1)

        self.residual = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2), # down sampling using avg pool
                                      conv2d_r)

        if equalized:
            self.block = nn.Sequential(
                EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, True),
                EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, True),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_features, in_features, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_features, out_features, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, True),
            )

        self.down_sample = nn.AvgPool2d(
            kernel_size=2, stride=2
        )  # down sampling using avg pool

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.scale = 1 / sqrt(2)

    def forward(self, x):
        residual = self.residual(x)

        x = self.block(x)
        x = self.down_sample(x)

        if self.dropout is None:
            return (x + residual) * self.scale
        else:
            return self.dropout((x + residual) * self.scale)