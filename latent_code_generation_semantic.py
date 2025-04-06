import torch


def get_w(batch_size, mapnet, w_dim, device, num_layers):
    z = torch.randn(batch_size, w_dim).to(device)
    w = mapnet(z)
    return w[None, :, :].expand(num_layers + 1, -1, -1)


def get_noise(batch_size, device, logres, num_layers, mask=False):
    noise = []
    res = 2 ** (logres - num_layers)

    for i in range(num_layers + 1):
        if not mask:
            if i == 0:
                n1 = None
            else:
                n1 = torch.randn(batch_size, 1, res, res, device=device)

            n2 = torch.randn(batch_size, 1, res, res, device=device)

            noise.append((n1, n2))

            res *= 2
        else:
            noise.append((None, None))

    return noise


def get_noise_custom(batch_size, device, logres, num_layers, mask=False):
    noise = []
    res = 2 ** (logres - num_layers + 1)

    for i in range(num_layers):
        if not mask:
            if i == 0:
                n1 = None
            else:
                n1 = torch.randn(batch_size, 1, res, res, device=device)

            n2 = torch.randn(batch_size, 1, res, res, device=device)

            noise.append((n1, n2))

            res *= 2
        else:
            noise.append((None, None))

    return noise