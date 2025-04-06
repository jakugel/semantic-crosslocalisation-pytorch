import torch
from latent_code_generation_semantic import get_w, get_noise, get_noise_custom
import numpy as np
from PIL import Image
import os
from math import sqrt


def generate_examples(gen_image, gen_mask, epoch, cross_level_test, log_resolution, num_layers, mapping_network_labelled, mapping_network_unlabelled,
                      w_dim, n=16, device='cpu', save_path="./crossloc_semantic_examples", train_images_l=True, train_masks=True,
                      train_images_u=True, train_cross=True, train_joint=True, use_custom_nets=False):
    gen_image.eval()
    gen_mask.eval()

    with torch.no_grad():
        if train_images_l:
            # generate labelled images
            w = get_w(n, mapping_network_labelled, w_dim, device, num_layers)
            if use_custom_nets:
                noise = get_noise_custom(n, device, log_resolution, num_layers)
            else:
                noise = get_noise(n, device, log_resolution, num_layers)

            img = gen_image(w, noise)

            img = img.detach().cpu().numpy()

            img = np.transpose(img, (0, 2, 3, 1))

            r = []

            for i in range(0, n, int(n / sqrt(n))):
                r.append(np.concatenate(img[i:i + int(n / sqrt(n))], axis=1))

            c1 = np.concatenate(r, axis=0)
            c1 = np.clip(c1, 0.0, 1.0)
            x = Image.fromarray(np.squeeze(np.uint8(c1 * 255)))

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            x.save(save_path + f'/epoch{epoch}.png')

        if train_images_u and train_cross:
            # generate cross/mixed images
            w_c_labelled = get_w(n,  mapping_network_labelled, w_dim, device, num_layers)
            w_c_unlabelled = get_w(n, mapping_network_unlabelled, w_dim, device, num_layers)

            tt = cross_level_test

            w_cross = torch.cat((w_c_labelled[:tt, :, :], w_c_unlabelled[:(num_layers - tt + 1), :, :]), axis=0)

            if use_custom_nets:
                noise = get_noise_custom(n, device, log_resolution, num_layers)
            else:
                noise = get_noise(n, device, log_resolution, num_layers)

            img = gen_image(w_cross, noise)

            img = img.detach().cpu().numpy()

            img = np.transpose(img, (0, 2, 3, 1))

            r = []

            for i in range(0, n, int(n / sqrt(n))):
                r.append(np.concatenate(img[i:i + int(n / sqrt(n))], axis=1))

            c1 = np.concatenate(r, axis=0)
            c1 = np.clip(c1, 0.0, 1.0)
            x = Image.fromarray(np.squeeze(np.uint8(c1 * 255)))

            x.save(save_path + f'/epoch{epoch}_c.png')

        if train_masks:
            # generate masks
            w_mask = get_w(n, mapping_network_labelled, w_dim, device, num_layers)

            if use_custom_nets:
                noise_mask = get_noise_custom(n, device, log_resolution, num_layers, mask=True)
            else:
                noise_mask = get_noise(n, device, log_resolution, num_layers, mask=True)

            img = gen_mask(w_mask, noise_mask)

            img = img.detach().cpu().numpy()

            img = np.transpose(img, (0, 2, 3, 1))

            r = []

            for i in range(0, n, int(n / sqrt(n))):
                r.append(np.concatenate(img[i:i + int(n / sqrt(n))], axis=1))

            c1 = np.concatenate(r, axis=0)
            c1 = np.clip(c1, 0.0, 1.0)
            x = Image.fromarray(np.squeeze(np.uint8(c1 * 255)))

            x.save(save_path + f'/epoch{epoch}_m.png')
