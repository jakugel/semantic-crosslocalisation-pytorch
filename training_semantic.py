from random import random
from synthesise_data_semantic import generate_examples
import torch
import numpy as np
from latent_code_generation_semantic import get_w, get_noise, get_noise_custom
from penalties_semantic import gradient_penalty
import os


def train_fn(
        critic_image, critic_mask, critic_joint,
        gen_image, gen_mask, mapnetl, mapnetu,
        path_length_penalty,
        images_labelled, masks, images_unlabelled,
        opt_critic_image, opt_critic_mask, opt_critic_joint,
        opt_gen_image, opt_gen_mask,
        opt_mapping_network_labelled, opt_mapping_network_unlabelled, num_epochs, batch_size, device,
        mixing_prob, logres, w_dim, lambda_gp, num_gen_layers, loss, train_images_l, train_masks, train_images_u,
        train_cross, train_joint, cross_level_train, cross_level_test, use_custom_nets, save_models,
        save_models_freq, save_models_dir, save_images, save_images_freq, save_images_dir
):

    if save_models:
        if not os.path.exists(save_models_dir):
            os.makedirs(save_models_dir)

    for epoch in range(num_epochs):
        if train_images_l:
            ########## LABELLED DATA ITERATION ############
            loss_critic_labelled, loss_gen_labelled = labelled_training_step(images_labelled, batch_size, device, mixing_prob, num_gen_layers,
                           mapnetl, w_dim, logres, gen_image, critic_image, loss, lambda_gp, epoch, opt_critic_image,
                           path_length_penalty, opt_gen_image, opt_mapping_network_labelled, use_custom_nets)

        if train_images_u:
            ########## UNLABELLED DATA ITERATION ############
            loss_critic_unlabelled, loss_gen_unlabelled = unlabelled_training_step(images_unlabelled, batch_size, device, mixing_prob, num_gen_layers, mapnetu, w_dim, logres,
                             gen_image, critic_image, loss, lambda_gp, opt_critic_image, epoch, path_length_penalty,
                             opt_gen_image, opt_mapping_network_unlabelled, use_custom_nets)

        if train_cross:
            ######## CROSS ITERATION ###########
            loss_critic_cross, loss_gen_cross = cross_training_step(images_unlabelled, batch_size, device, num_gen_layers,
                        mapnetl, w_dim, mapnetu, logres, gen_image, critic_image, loss, lambda_gp, opt_critic_image,
                        epoch, path_length_penalty, opt_gen_image, opt_mapping_network_unlabelled, opt_mapping_network_labelled,
                                                                    cross_level_train, use_custom_nets)

        if train_masks:
            ######## MASK ITERATION ###########
            loss_critic_mask, loss_gen_mask = mask_training_step(masks, batch_size, device, mixing_prob, num_gen_layers,
                               mapnetl, w_dim, logres, gen_mask, critic_mask, loss, lambda_gp, epoch, opt_critic_mask,
                               path_length_penalty, opt_gen_mask, opt_mapping_network_labelled, use_custom_nets)

        if train_joint:
            ######## JOINT ITERATION ###########
            loss_critic_joint, loss_gen_joint = joint_training_step(images_labelled, masks, batch_size, device, mixing_prob, num_gen_layers,
                               mapnetl, w_dim, logres, gen_image, gen_mask, critic_joint, loss, lambda_gp, epoch, opt_critic_joint,
                               path_length_penalty, opt_gen_image, opt_gen_mask, opt_mapping_network_labelled, use_custom_nets)

        ######### Print losses every 10 epochs ##########
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}")
            if train_images_l:
                print(f"Loss Critic Labelled: {loss_critic_labelled.item():.4f}")
                print(f"Loss Gen Labelled: {loss_gen_labelled.item():.4f}")
            if train_images_u:
                print(f"Loss Critic Unlabelled: {loss_critic_unlabelled.item():.4f}")
                print(f"Loss Gen Unlabelled: {loss_gen_unlabelled.item():.4f}")
            if train_cross:
                print(f"Loss Critic Cross: {loss_critic_cross.item():.4f}")
                print(f"Loss Gen Cross: {loss_gen_cross.item():.4f}")
            if train_masks:
                print(f"Loss Critic Mask: {loss_critic_mask.item():.4f}")
                print(f"Loss Gen Mask: {loss_gen_mask.item():.4f}")
            if train_joint:
                print(f"Loss Critic Joint: {loss_critic_joint.item():.4f}")
                print(f"Loss Gen Joint: {loss_gen_joint.item():.4f}")
            print("-" * 50)

        if epoch % save_images_freq == 0 and save_images:
            generate_examples(gen_image, gen_mask, epoch, cross_level_test, log_resolution=logres, num_layers=num_gen_layers, mapping_network_labelled=mapnetl,
                              mapping_network_unlabelled=mapnetu, w_dim=w_dim, device=device, train_images_l=train_images_l,
                              train_masks=train_masks, train_images_u=train_images_u, train_cross=train_cross,
                              train_joint=train_joint, use_custom_nets=use_custom_nets, save_path=save_images_dir)

            gen_image.train()
            gen_mask.train()

        if epoch % save_models_freq == 0 and save_models:
            torch.save(critic_image.state_dict(), save_models_dir + f"/critici_{epoch}.pth")
            print('saved model at ' + save_models_dir + f"/critici_{epoch}.pth")
            torch.save(critic_mask.state_dict(), save_models_dir + f"/criticm_{epoch}.pth")
            print('saved model at ' + save_models_dir + f"/criticm_{epoch}.pth")
            torch.save(critic_joint.state_dict(), save_models_dir + f"/criticj_{epoch}.pth")
            print('saved model at ' + save_models_dir + f"/criticj_{epoch}.pth")
            torch.save(gen_image.state_dict(), save_models_dir + f"/geni_{epoch}.pth")
            print('saved model at ' + save_models_dir + f"/geni_{epoch}.pth")
            torch.save(gen_mask.state_dict(), save_models_dir + f"/genm_{epoch}.pth")
            print('saved model at ' + save_models_dir + f"/genm_{epoch}.pth")
            torch.save(mapnetl.state_dict(), save_models_dir + f"/mapnetl_{epoch}.pth")
            print('saved model at ' + save_models_dir + f"/mapnetl_{epoch}.pth")
            torch.save(mapnetu.state_dict(), save_models_dir + f"/mapnetu_{epoch}.pth")
            print('saved model at ' + save_models_dir + f"/mapnetu_{epoch}.pth")


def labelled_training_step(images_labelled, batch_size, device, mixing_prob, num_gen_layers,
                           mapnetl, w_dim, logres, gen_image, critic_image, loss, lambda_gp, epoch, opt_critic_image,
                           path_length_penalty, opt_gen_image, opt_mapping_network_labelled, use_custom_nets):
    # get real image batch
    idx1 = np.random.randint(0, images_labelled.shape[0], batch_size)

    # images
    batch_images_l = images_labelled[idx1].astype('float32') / 255.0

    real_labelled = torch.Tensor(batch_images_l)
    real_labelled = real_labelled.to(device)

    cur_batch_size = real_labelled.shape[0]

    # style mixing
    if random() < mixing_prob:
        # mixing
        w1_labelled = get_w(cur_batch_size, device=device, num_layers=num_gen_layers, mapnet=mapnetl,
                                     w_dim=w_dim)
        w2_labelled = get_w(cur_batch_size, device=device, num_layers=num_gen_layers, mapnet=mapnetl,
                                     w_dim=w_dim)

        tt = int(random() * num_gen_layers)  # style level to crossover

        w_labelled = torch.cat((w1_labelled[:tt, :, :], w2_labelled[:(num_gen_layers - tt + 1), :, :]), axis=0)
    else:
        # no mixing
        w_labelled = get_w(cur_batch_size, device=device, num_layers=num_gen_layers, mapnet=mapnetl,
                                    w_dim=w_dim)

    if use_custom_nets:
        noise = get_noise_custom(cur_batch_size, device=device, logres=logres, num_layers=num_gen_layers)
    else:
        noise = get_noise(cur_batch_size, device=device, logres=logres, num_layers=num_gen_layers)

    fake_labelled = gen_image(w_labelled, noise)  # get fake labelled images
    critic_fake_labelled, _ = critic_image(fake_labelled.detach())  # compute critic scores for fake images

    critic_real_labelled, _ = critic_image(real_labelled)  # compute critic scores for real images

    # compute gradient penalty
    gp_labelled = gradient_penalty(critic_image, real_labelled, fake_labelled, device=device)

    # compute full critic loss using labelled image training
    if loss == "wgan":
        loss_critic_labelled = (
                -(torch.mean(critic_real_labelled) - torch.mean(critic_fake_labelled))
                + lambda_gp * gp_labelled
                + (0.001 * torch.mean(critic_real_labelled ** 2))
        )
    elif loss == "hinge":
        loss_critic_labelled = (
                torch.mean(torch.relu(1 + critic_real_labelled) + torch.relu(1 - critic_fake_labelled))
                + lambda_gp * gp_labelled
        )

    critic_image.zero_grad()
    loss_critic_labelled.backward()
    opt_critic_image.step()

    gen_fake_labelled, _ = critic_image(fake_labelled)

    if loss == "wgan":
        loss_gen_labelled = -torch.mean(gen_fake_labelled)
    elif loss == "hinge":
        loss_gen_labelled = torch.mean(gen_fake_labelled)

    if epoch % 16 == 0:
        plp_labelled = path_length_penalty(w_labelled, fake_labelled)
        if not torch.isnan(plp_labelled):
            loss_gen_labelled = loss_gen_labelled + plp_labelled

    mapnetl.zero_grad()
    gen_image.zero_grad()
    loss_gen_labelled.backward()
    opt_gen_image.step()
    opt_mapping_network_labelled.step()

    return loss_critic_labelled, loss_gen_labelled


def unlabelled_training_step(images_unlabelled, batch_size, device, mixing_prob, num_gen_layers, mapnetu, w_dim, logres,
                             gen_image, critic_image, loss, lambda_gp, opt_critic_image, epoch, path_length_penalty,
                             opt_gen_image, opt_mapping_network_unlabelled, use_custom_nets):
    idx2 = np.random.randint(0, images_unlabelled.shape[0], batch_size)

    batch_images_u = images_unlabelled[idx2].astype('float32') / 255.0

    real_unlabelled = torch.Tensor(batch_images_u)
    real_unlabelled = real_unlabelled.to(device)
    cur_batch_size = real_unlabelled.shape[0]

    if random() < mixing_prob:
        w1_unlabelled = get_w(cur_batch_size, device=device, num_layers=num_gen_layers, mapnet=mapnetu,
                                         w_dim=w_dim)
        w2_unlabelled = get_w(cur_batch_size, device=device, num_layers=num_gen_layers, mapnet=mapnetu,
                                         w_dim=w_dim)

        tt = int(random() * num_gen_layers)

        w_unlabelled = torch.cat((w1_unlabelled[:tt, :, :], w2_unlabelled[:(num_gen_layers - tt + 1), :, :]), axis=0)

    else:
        w_unlabelled = get_w(cur_batch_size, device=device, num_layers=num_gen_layers, mapnet=mapnetu,
                                        w_dim=w_dim)

    if use_custom_nets:
        noise = get_noise_custom(cur_batch_size, device=device, logres=logres, num_layers=num_gen_layers)
    else:
        noise = get_noise(cur_batch_size, device=device, logres=logres, num_layers=num_gen_layers)

    fake_unlabelled = gen_image(w_unlabelled, noise)
    _, critic_fake_unlabelled = critic_image(fake_unlabelled.detach())
    _, critic_real_unlabelled = critic_image(real_unlabelled)
    gp_unlabelled = gradient_penalty(critic_image, real_unlabelled, fake_unlabelled, labelled=False, device=device)

    if loss == "wgan":
        loss_critic_unlabelled = (
                -(torch.mean(critic_real_unlabelled) - torch.mean(critic_fake_unlabelled))
                + lambda_gp * gp_unlabelled
                + (0.001 * torch.mean(critic_real_unlabelled ** 2))
        )
    elif loss == "hinge":
        loss_critic_unlabelled = (
                torch.mean(torch.relu(1 + critic_real_unlabelled) + torch.relu(1 - critic_fake_unlabelled))
                + lambda_gp * gp_unlabelled
        )

    critic_image.zero_grad()
    loss_critic_unlabelled.backward()
    opt_critic_image.step()

    _, gen_fake_unlabelled = critic_image(fake_unlabelled)

    if loss == "wgan":
        loss_gen_unlabelled = -torch.mean(gen_fake_unlabelled)
    elif loss == "hinge":
        loss_gen_unlabelled = torch.mean(gen_fake_unlabelled)

    if epoch % 16 == 0:
        plp_unlabelled = path_length_penalty(w_unlabelled, fake_unlabelled)
        if not torch.isnan(plp_unlabelled):
            loss_gen_unlabelled = loss_gen_unlabelled + plp_unlabelled

    mapnetu.zero_grad()
    gen_image.zero_grad()
    loss_gen_unlabelled.backward()
    opt_gen_image.step()
    opt_mapping_network_unlabelled.step()

    return loss_critic_unlabelled, loss_gen_unlabelled


def cross_training_step(images_unlabelled, batch_size, device, num_gen_layers,
                        mapnetl, w_dim, mapnetu, logres, gen_image, critic_image, loss, lambda_gp, opt_critic_image,
                        epoch, path_length_penalty, opt_gen_image, opt_mapping_network_unlabelled, opt_mapping_network_labelled,
                        cross_level_train, use_custom_nets):
    idx3 = np.random.randint(0, images_unlabelled.shape[0], batch_size)

    batch_images_c = images_unlabelled[idx3].astype('float32') / 255.0

    real_cross = torch.Tensor(batch_images_c)
    real_cross = real_cross.to(device)

    cur_batch_size = real_cross.shape[0]

    w_c_labelled = get_w(cur_batch_size, device=device, num_layers=num_gen_layers, mapnet=mapnetl,
                                  w_dim=w_dim)
    w_c_unlabelled = get_w(cur_batch_size, device=device, num_layers=num_gen_layers, mapnet=mapnetu,
                                      w_dim=w_dim)

    tt = cross_level_train

    w_cross = torch.cat((w_c_labelled[:tt, :, :], w_c_unlabelled[:(num_gen_layers - tt + 1), :, :]), axis=0)

    if use_custom_nets:
        noise = get_noise_custom(cur_batch_size, device=device, logres=logres, num_layers=num_gen_layers)
    else:
        noise = get_noise(cur_batch_size, device=device, logres=logres, num_layers=num_gen_layers)

    fake_cross = gen_image(w_cross, noise)
    _, critic_fake_cross = critic_image(fake_cross.detach())

    _, critic_real_cross = critic_image(real_cross)
    gp_cross = gradient_penalty(critic_image, real_cross, fake_cross, labelled=False, device=device)

    if loss == 'wgan':
        loss_critic_cross = (
                -(torch.mean(critic_real_cross) - torch.mean(critic_fake_cross))
                + lambda_gp * gp_cross
                + (0.001 * torch.mean(critic_real_cross ** 2))
        )
    elif loss == 'hinge':
        loss_critic_cross = (
                torch.mean(torch.relu(1 + critic_real_cross) + torch.relu(1 - critic_fake_cross))
                + lambda_gp * gp_cross
        )

    critic_image.zero_grad()
    loss_critic_cross.backward()
    opt_critic_image.step()

    _, gen_fake_cross = critic_image(fake_cross)

    if loss == 'wgan':
        loss_gen_cross = -torch.mean(gen_fake_cross)
    elif loss == 'hinge':
        loss_gen_cross = torch.mean(gen_fake_cross)

    if epoch % 16 == 0:
        plp_cross = path_length_penalty(w_cross, fake_cross)
        if not torch.isnan(plp_cross):
            loss_gen_cross = loss_gen_cross + plp_cross

    mapnetl.zero_grad()
    mapnetu.zero_grad()
    gen_image.zero_grad()
    loss_gen_cross.backward()
    opt_gen_image.step()
    opt_mapping_network_labelled.step()
    opt_mapping_network_unlabelled.step()

    return loss_critic_cross, loss_gen_cross


def mask_training_step(masks, batch_size, device, mixing_prob, num_gen_layers,
                           mapnetl, w_dim, logres, gen_mask, critic_mask, loss, lambda_gp, epoch, opt_critic_mask,
                           path_length_penalty, opt_gen_mask, opt_mapping_network_labelled, use_custom_nets):
    # get real image batch
    idx1 = np.random.randint(0, masks.shape[0], batch_size)

    # images
    batch_masks = masks[idx1].astype('float32') * (255.0 / 3) / 255.0

    real_masks = torch.Tensor(batch_masks)
    real_masks = real_masks.to(device)

    cur_batch_size = real_masks.shape[0]

    # style mixing
    if random() < mixing_prob:
        # mixing
        w1_labelled = get_w(cur_batch_size, device=device, num_layers=num_gen_layers, mapnet=mapnetl,
                                     w_dim=w_dim)
        w2_labelled = get_w(cur_batch_size, device=device, num_layers=num_gen_layers, mapnet=mapnetl,
                                     w_dim=w_dim)

        tt = int(random() * num_gen_layers)  # style level to crossover

        w_labelled = torch.cat((w1_labelled[:tt, :, :], w2_labelled[:(num_gen_layers - tt + 1), :, :]), axis=0)
    else:
        # no mixing
        w_labelled = get_w(cur_batch_size, device=device, num_layers=num_gen_layers, mapnet=mapnetl,
                                    w_dim=w_dim)

    if use_custom_nets:
        noise = get_noise_custom(cur_batch_size, device=device, logres=logres, num_layers=num_gen_layers, mask=True)
    else:
        noise = get_noise(cur_batch_size, device=device, logres=logres, num_layers=num_gen_layers, mask=True)

    fake_masks = gen_mask(w_labelled, noise)  # get fake labelled images
    critic_fake_mask = critic_mask(fake_masks.detach())  # compute critic scores for fake images

    critic_real_mask = critic_mask(real_masks)  # compute critic scores for real images

    # compute gradient penalty
    gp_mask = gradient_penalty(critic_mask, real_masks, fake_masks, mask=True, device=device)

    # compute full critic loss using labelled image training
    if loss == "wgan":
        loss_critic_mask = (
                -(torch.mean(critic_real_mask) - torch.mean(critic_fake_mask))
                + lambda_gp * gp_mask
                + (0.001 * torch.mean(critic_real_mask ** 2))
        )
    elif loss == "hinge":
        loss_critic_mask = (
                torch.mean(torch.relu(1 + critic_real_mask) + torch.relu(1 - critic_fake_mask))
                + lambda_gp * gp_mask
        )

    critic_mask.zero_grad()
    loss_critic_mask.backward()
    opt_critic_mask.step()

    gen_fake_labelled = critic_mask(fake_masks)

    if loss == "wgan":
        loss_gen_mask = -torch.mean(gen_fake_labelled)
    elif loss == "hinge":
        loss_gen_mask = torch.mean(gen_fake_labelled)

    if epoch % 16 == 0:
        plp_labelled = path_length_penalty(w_labelled, fake_masks)
        if not torch.isnan(plp_labelled):
            loss_gen_mask = loss_gen_mask + plp_labelled

    mapnetl.zero_grad()
    gen_mask.zero_grad()
    loss_gen_mask.backward()
    opt_gen_mask.step()
    opt_mapping_network_labelled.step()

    return loss_critic_mask, loss_gen_mask


def joint_training_step(images_labelled, masks, batch_size, device, mixing_prob, num_gen_layers,
                           mapnetl, w_dim, logres, gen_image, gen_mask, critic_joint, loss, lambda_gp, epoch, opt_critic_joint,
                           path_length_penalty, opt_gen_image, opt_gen_mask, opt_mapping_network_labelled, use_custom_nets):
    # get real image batch
    idx1 = np.random.randint(0, images_labelled.shape[0], batch_size)

    # images
    batch_images_l = images_labelled[idx1].astype('float32') / 255.0
    batch_masks = masks[idx1].astype('float32') * (255.0 / 3) / 255.0

    real_labelled = torch.Tensor(batch_images_l)
    real_labelled = real_labelled.to(device)

    real_masks = torch.Tensor(batch_masks)
    real_masks = real_masks.to(device)

    cur_batch_size = real_labelled.shape[0]

    # style mixing
    if random() < mixing_prob:
        # mixing
        w1_labelled = get_w(cur_batch_size, device=device, num_layers=num_gen_layers, mapnet=mapnetl,
                                     w_dim=w_dim)
        w2_labelled = get_w(cur_batch_size, device=device, num_layers=num_gen_layers, mapnet=mapnetl,
                                     w_dim=w_dim)

        tt = int(random() * num_gen_layers)  # style level to crossover

        w_labelled = torch.cat((w1_labelled[:tt, :, :], w2_labelled[:(num_gen_layers - tt + 1), :, :]), axis=0)
    else:
        # no mixing
        w_labelled = get_w(cur_batch_size, device=device, num_layers=num_gen_layers, mapnet=mapnetl,
                                    w_dim=w_dim)

    if use_custom_nets:
        noise = get_noise_custom(cur_batch_size, device=device, logres=logres, num_layers=num_gen_layers)
        noise_mask = get_noise_custom(cur_batch_size, device=device, logres=logres, num_layers=num_gen_layers, mask=True)
    else:
        noise = get_noise(cur_batch_size, device=device, logres=logres, num_layers=num_gen_layers)
        noise_mask = get_noise(cur_batch_size, device=device, logres=logres, num_layers=num_gen_layers, mask=True)

    fake_labelled = gen_image(w_labelled, noise)  # get fake labelled images
    fake_masks = gen_mask(w_labelled, noise_mask)   # get corresponding fake masks

    critic_fake_joint = critic_joint(fake_labelled.detach(), fake_masks.detach())  # compute critic scores for fake images

    critic_real_joint = critic_joint(real_labelled, real_masks)  # compute critic scores for real images

    # compute gradient penalty
    gp_joint = gradient_penalty(critic_joint, [real_labelled, real_masks], [fake_labelled, fake_masks], joint=True, device=device)

    # compute full critic loss using labelled image training
    if loss == "wgan":
        loss_critic_joint = (
                -(torch.mean(critic_real_joint) - torch.mean(critic_fake_joint))
                + lambda_gp * gp_joint
                + (0.001 * torch.mean(critic_real_joint ** 2))
        )
    elif loss == "hinge":
        loss_critic_joint = (
                torch.mean(torch.relu(1 + critic_real_joint) + torch.relu(1 - critic_fake_joint))
                + lambda_gp * gp_joint
        )

    critic_joint.zero_grad()
    loss_critic_joint.backward()
    opt_critic_joint.step()

    gen_fake_joint = critic_joint(fake_labelled, fake_masks)

    if loss == "wgan":
        loss_gen_joint = -torch.mean(gen_fake_joint)
    elif loss == "hinge":
        loss_gen_joint = torch.mean(gen_fake_joint)

    if epoch % 16 == 0:
        plp_labelled = path_length_penalty(w_labelled, fake_labelled)
        plp_masks = path_length_penalty(w_labelled, fake_masks)

        if not torch.isnan(plp_labelled):
            loss_gen_joint = loss_gen_joint + (plp_labelled + plp_masks) / 2

    mapnetl.zero_grad()
    gen_image.zero_grad()
    gen_mask.zero_grad()
    loss_gen_joint.backward()
    opt_gen_image.step()
    opt_gen_mask.step()
    opt_mapping_network_labelled.step()

    return loss_critic_joint, loss_gen_joint