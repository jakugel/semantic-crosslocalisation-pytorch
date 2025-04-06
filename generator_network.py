import torch.nn as nn
import torch.nn.functional as F
import torch
from utility_layers import EqualizedWeight, Weight, EqualizedLinear


class Generator(nn.Module):

    def __init__(self, log_resolution, W_DIM, n_features = 32, max_features = 256, num_layers = 4, equalized=True):

        super().__init__()

        features = [min(max_features, n_features * (2 ** i)) for i in range(num_layers, -1, -1)]
        self.n_blocks = len(features)

        initial_resolution = log_resolution - self.n_blocks + 1
        initial_size = 2 ** initial_resolution

        self.initial_constant = nn.Parameter(torch.randn((1, features[0], initial_size, initial_size)))

        self.style_block = StyleBlock(W_DIM, features[0], features[0], equalized=equalized)
        self.to_rgb = ToRGB(W_DIM, features[0], equalized=equalized)

        blocks = [GeneratorBlock(W_DIM, features[i - 1], features[i], equalized=equalized) for i in range(1, self.n_blocks)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, w, input_noise):

        batch_size = w.shape[1]

        x = self.initial_constant.expand(batch_size, -1, -1, -1)
        x = self.style_block(x, w[0], input_noise[0][1])
        rgb = self.to_rgb(x, w[0])

        for i in range(1, self.n_blocks):
            x = F.interpolate(x, scale_factor=2, mode="bilinear")
            x, rgb_new = self.blocks[i - 1](x, w[i], input_noise[i])
            rgb = F.interpolate(rgb, scale_factor=2, mode="bilinear") + rgb_new

        return torch.tanh(rgb)


class GeneratorBlock(nn.Module):

    def __init__(self, W_DIM, in_features, out_features, equalized=True):

        super().__init__()

        self.style_block1 = StyleBlock(W_DIM, in_features, out_features, equalized=equalized)
        self.style_block2 = StyleBlock(W_DIM, out_features, out_features, equalized=equalized)

        self.to_rgb = ToRGB(W_DIM, out_features, equalized=equalized)

    def forward(self, x, w, noise):

        x = self.style_block1(x, w, noise[0])
        x = self.style_block2(x, w, noise[1])

        rgb = self.to_rgb(x, w)

        return x, rgb


class StyleBlock(nn.Module):

    def __init__(self, W_DIM, in_features, out_features, equalized=True):

        super().__init__()

        if equalized:
            self.to_style = EqualizedLinear(W_DIM, in_features, bias=1.0)
        else:
            self.to_style = nn.Linear(W_DIM, in_features, bias=True)

        self.conv = Conv2dWeightModulate(in_features, out_features, kernel_size=3, equalized=equalized)
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w, noise):
        s = self.to_style(w)
        x = self.conv(x, s)
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        return self.activation(x + self.bias[None, :, None, None])


class ToRGB(nn.Module):

    def __init__(self, W_DIM, features, equalized=True):

        super().__init__()
        if equalized:
            self.to_style = EqualizedLinear(W_DIM, features, bias=1.0)
        else:
            self.to_style = nn.Linear(W_DIM, features, bias=True)

        self.conv = Conv2dWeightModulate(features, 1, kernel_size=1, demodulate=False, equalized=equalized)
        self.bias = nn.Parameter(torch.zeros(1))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w):
        style = self.to_style(w)
        x = self.conv(x, style)
        return self.activation(x + self.bias[None, :, None, None])


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