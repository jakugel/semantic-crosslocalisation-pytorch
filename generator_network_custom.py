import torch.nn as nn
import torch.nn.functional as F
import torch
from utility_layers import EqualizedWeight, Weight, EqualizedLinear


class GeneratorCustom(nn.Module):

    def __init__(self, log_resolution, W_DIM, n_features = 32, num_layers = 4, equalized=True):

        super().__init__()

        features = [n_features for _ in range(num_layers)]
        self.n_blocks = len(features)

        initial_resolution = log_resolution - self.n_blocks + 1
        initial_size = 2 ** initial_resolution

        self.initial_constant = nn.Parameter(torch.randn((1, features[0], initial_size, initial_size)))
        self.initial_act = nn.ReLU()

        blocks = [GeneratorBlock(W_DIM, features[0], features[0], img_size=2**log_resolution,
                                 equalized=equalized, upsample=False)] + \
            [GeneratorBlock(W_DIM, features[i - 1], features[i], img_size=2**log_resolution,
                            equalized=equalized) for i in range(1, self.n_blocks)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, w, input_noise):

        batch_size = w.shape[1]

        outs = []

        x = self.initial_constant.expand(batch_size, -1, -1, -1)
        x = self.initial_act(x)

        for i in range(self.n_blocks):
            x, rgb_new = self.blocks[i](x, w[i], input_noise[i])
            outs.append(rgb_new)

        x = torch.sum(torch.stack(outs), dim=0)

        return x / 2 + 0.5


class GeneratorBlock(nn.Module):

    def __init__(self, W_DIM, in_features, out_features, img_size, equalized=True, upsample=True):

        super().__init__()

        self.upsample = upsample

        if equalized:
            self.to_style = EqualizedLinear(W_DIM, in_features)
        else:
            self.to_style = nn.Linear(W_DIM, in_features)

        self.style_block1 = StyleBlock(in_features, out_features, equalized=equalized)
        self.style_block2 = StyleBlock(out_features, out_features, equalized=equalized)

        self.to_rgb = ToRGB(out_features, img_size= img_size, equalized=equalized)

    def forward(self, x, w, noise):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear")

        s = self.to_style(w)
        x = self.style_block1(x, s, noise[0])
        x = self.style_block2(x, s, noise[1])

        rgb = self.to_rgb(x, s)

        return x, rgb


class StyleBlock(nn.Module):
    def __init__(self, in_features, out_features, equalized=True):
        super().__init__()

        self.conv = Conv2dWeightModulate(in_features, out_features, kernel_size=3, equalized=equalized)

        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, s, noise):
        x = self.conv(x, s)
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        return self.activation(x + self.bias[None, :, None, None])


class ToRGB(nn.Module):
    def __init__(self, features, img_size, equalized=True):
        super().__init__()
        self.conv = Conv2dWeightModulate(features, 1, kernel_size=1, demodulate=False, equalized=equalized)
        self.img_size = img_size

    def forward(self, x, style):
        x = self.conv(x, style)

        scale_factor = int(self.img_size / x.shape[-1])
        rgb = F.interpolate(x, scale_factor=scale_factor, mode="bilinear")  # interpolate, resizing to full size

        return rgb


class Conv2dWeightModulate(nn.Module):
    def __init__(self, in_features, out_features, kernel_size,
                 demodulate = True, eps = 1e-8, equalized=True):
        super().__init__()
        self.out_features = out_features
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2
        if equalized:
            self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        else:
            self.weight = Weight([out_features, in_features, kernel_size, kernel_size])

        self.eps = eps

    def forward(self, x, s):
        b, _, h, w = x.shape

        s = s[:, None, :, None, None]
        weights = self.weight()[None, :, :, :, :]
        weights = weights * s

        if self.demodulate:
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * sigma_inv

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        x = F.conv2d(x, weights, padding=self.padding, groups=b)

        return x.reshape(-1, self.out_features, h, w)