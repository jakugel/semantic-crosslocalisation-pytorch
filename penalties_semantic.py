import torch.nn as nn
import torch
from math import sqrt


class PathLengthPenalty(nn.Module):

    def __init__(self, beta):

        super().__init__()

        self.beta = beta
        self.steps = nn.Parameter(torch.tensor(0.), requires_grad=False)

        self.exp_sum_a = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, w, x):

        device = x.device
        image_size = x.shape[2] * x.shape[3]
        y = torch.randn(x.shape, device=device)

        output = (x * y).sum() / sqrt(image_size)
        sqrt(image_size)

        gradients, *_ = torch.autograd.grad(outputs=output,
                                            inputs=w,
                                            grad_outputs=torch.ones(output.shape, device=device),
                                            create_graph=True)

        norm = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()

        if self.steps > 0:

            a = self.exp_sum_a / (1 - self.beta ** self.steps)

            loss = torch.mean((norm - a) ** 2)
        else:
            loss = norm.new_tensor(0)

        mean = norm.mean().detach()
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)
        self.steps.add_(1.)

        return loss


def gradient_penalty(critic, real, fake, labelled=True, mask=False, joint=False, device="cpu"):
    if not joint:
        BATCH_SIZE, C, H, W = real.shape
    else:
        BATCH_SIZE, C, H, W = real[0].shape

    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)

    if not joint:
        interpolated_images = real * beta + fake.detach() * (1 - beta)
    else:
        interpolated_images = real[0] * beta + fake[0].detach() * (1 - beta)
        interpolated_masks = real[1] * beta + fake[1].detach() * (1 - beta)
        interpolated_masks.requires_grad_(True)

    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    if labelled and not mask and not joint:
        mixed_scores, _ = critic(interpolated_images)
    elif not labelled and not joint:
        _, mixed_scores = critic(interpolated_images)
    elif mask and not joint:
        mixed_scores = critic(interpolated_images)
    elif joint:
        mixed_scores = critic(interpolated_images, interpolated_masks)

    # Take the gradient of the scores with respect to the images
    if joint:
        inputs = [interpolated_images, interpolated_masks]
    else:
        inputs = interpolated_images

    gradient = torch.autograd.grad(
        inputs=inputs,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty