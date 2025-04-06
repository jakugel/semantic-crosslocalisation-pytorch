import torch
from data_loaders_semantic import get_data_labelled, get_data_unlabelled
from generator_network import Generator
from generator_network_custom import GeneratorCustom
from discriminator_networks_semantic import DiscriminatorImage, DiscriminatorMask, DiscriminatorJoint
from discriminator_networks_semantic_custom import DiscriminatorImageCustom, DiscriminatorMaskCustom, DiscriminatorJointCustom
from mapping_networks_semantic import MappingNetwork
from mapping_networks_semantic_custom import MappingNetworkCustom
from penalties import PathLengthPenalty
from torch import optim
from training_semantic import train_fn
from math import log2

# Generator parameters
NUM_GEN_LAYERS = 5
START_GEN_FEATURES = 32
MAX_GEN_FEATURES = 256

# Discriminator parameters
NUM_DIS_LAYERS = 5
START_DIS_FEATURES = 64
MAX_DIS_FEATURES = 256
DIS_DROPOUT = 0.25

# Minibatch discrimination parameters
MDL_KERNELS = 50
MDL_KERNEL_SIZE = 30

# Mapping network parameters
NUM_MAP_LAYERS = 4
MAPPING_LR_MULT = 0.1
W_DIM = 128
Z_DIM = 128
MAP_LEAKY = 0.2

# Other parameters
LR = 5e-5
ADAM_B1 = 0.0
ADAM_B2 = 0.999
NUM_CLASSES = 4
MIXING_PROB = 0.9
BATCH_SIZE = 8
LAMBDA_GP = 10
PL_BETA = 0.99
IMG_SIZE = 128
LOG_RESOLUTION = int(log2(IMG_SIZE))
NUM_EPOCHS = 100000
H5FILEPATH_LABELLED = "./data/your_labelled_data.hdf5"
H5FILEPATH_UNLABELLED = "./data/your_unlabelled_data.hdf5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOSS_FN = "hinge"    # "wgan", "hinge"
EQUALIZED = False    # whether to use equalized layers
CROSS_LEVEL_TRAIN = 1
CROSS_LEVEL_TEST = NUM_GEN_LAYERS - 1
USE_CUSTOM_NETS = True
SAVE_MODELS = True
SAVE_MODELS_FREQ = 1000
SAVE_MODELS_DIR = "./crossloc_semantic_models"
SAVE_IMAGES = True
SAVE_IMAGES_FREQ = 100
SAVE_IMAGES_DIR = "./crossloc_semantic_images"

TRAIN_IMAGES_L = True
TRAIN_MASKS = True
TRAIN_IMAGES_U = True
TRAIN_CROSS = True
TRAIN_JOINT = True

# setup networks
if USE_CUSTOM_NETS:
    geni = GeneratorCustom(LOG_RESOLUTION, W_DIM, START_GEN_FEATURES, NUM_GEN_LAYERS, EQUALIZED).to(DEVICE)
    genm = GeneratorCustom(LOG_RESOLUTION, W_DIM, START_GEN_FEATURES, NUM_GEN_LAYERS, EQUALIZED).to(DEVICE)

    critici = DiscriminatorImageCustom(LOG_RESOLUTION, START_DIS_FEATURES, NUM_DIS_LAYERS,
                                 MDL_KERNELS, MDL_KERNEL_SIZE, EQUALIZED, DIS_DROPOUT).to(DEVICE)
    criticm = DiscriminatorMaskCustom(LOG_RESOLUTION, START_DIS_FEATURES, NUM_DIS_LAYERS,
                                MDL_KERNELS, MDL_KERNEL_SIZE, EQUALIZED, DIS_DROPOUT).to(DEVICE)
    criticj = DiscriminatorJointCustom(LOG_RESOLUTION, START_DIS_FEATURES, NUM_DIS_LAYERS,
                                 MDL_KERNELS, MDL_KERNEL_SIZE, EQUALIZED, DIS_DROPOUT).to(DEVICE)

    mapnetl = MappingNetworkCustom(Z_DIM, W_DIM, NUM_MAP_LAYERS, EQUALIZED, MAP_LEAKY).to(DEVICE)
    mapnetu = MappingNetworkCustom(Z_DIM, W_DIM, NUM_MAP_LAYERS, EQUALIZED, MAP_LEAKY).to(DEVICE)
else:
    geni = Generator(LOG_RESOLUTION, W_DIM, START_GEN_FEATURES, MAX_GEN_FEATURES, NUM_GEN_LAYERS, EQUALIZED).to(DEVICE)
    genm = Generator(LOG_RESOLUTION, W_DIM, START_GEN_FEATURES, MAX_GEN_FEATURES, NUM_GEN_LAYERS, EQUALIZED).to(DEVICE)

    critici = DiscriminatorImage(LOG_RESOLUTION, START_DIS_FEATURES, MAX_DIS_FEATURES, NUM_DIS_LAYERS,
                                 MDL_KERNELS, MDL_KERNEL_SIZE, EQUALIZED, DIS_DROPOUT).to(DEVICE)
    criticm = DiscriminatorMask(LOG_RESOLUTION, START_DIS_FEATURES, MAX_DIS_FEATURES, NUM_DIS_LAYERS,
                                MDL_KERNELS, MDL_KERNEL_SIZE, EQUALIZED, DIS_DROPOUT).to(DEVICE)
    criticj = DiscriminatorJoint(LOG_RESOLUTION, START_DIS_FEATURES, MAX_DIS_FEATURES, NUM_DIS_LAYERS,
                                 MDL_KERNELS, MDL_KERNEL_SIZE, EQUALIZED, DIS_DROPOUT).to(DEVICE)

    mapnetl = MappingNetwork(Z_DIM, W_DIM, NUM_MAP_LAYERS, EQUALIZED, MAP_LEAKY).to(DEVICE)
    mapnetu = MappingNetwork(Z_DIM, W_DIM, NUM_MAP_LAYERS, EQUALIZED, MAP_LEAKY).to(DEVICE)

images_l, masks = get_data_labelled(H5FILEPATH_LABELLED)
images_u = get_data_unlabelled(H5FILEPATH_UNLABELLED)

opt_critici = optim.Adam(critici.parameters(), lr=LR, betas=(ADAM_B1, ADAM_B2))
opt_criticm = optim.Adam(criticm.parameters(), lr=LR, betas=(ADAM_B1, ADAM_B2))
opt_criticj = optim.Adam(criticm.parameters(), lr=LR, betas=(ADAM_B1, ADAM_B2))
opt_mapnetl = optim.Adam(mapnetl.parameters(), lr=LR * MAPPING_LR_MULT, betas=(ADAM_B1, ADAM_B2))
opt_mapnetu = optim.Adam(mapnetu.parameters(), lr=LR * MAPPING_LR_MULT, betas=(ADAM_B1, ADAM_B2))
opt_geni = optim.Adam(geni.parameters(), lr=LR, betas=(ADAM_B1, ADAM_B2))
opt_genm = optim.Adam(genm.parameters(), lr=LR, betas=(ADAM_B1, ADAM_B2))

critici.train()
criticm.train()
criticj.train()
mapnetl.train()
mapnetu.train()
geni.train()
genm.train()

path_length_penalty = PathLengthPenalty(PL_BETA).to(DEVICE)

train_fn(
    critici, criticm, criticj,
    geni, genm, mapnetl, mapnetu,
    path_length_penalty,
    images_l, masks, images_u,
    opt_critici, opt_criticm, opt_criticj,
    opt_geni, opt_genm,
    opt_mapnetl, opt_mapnetu, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE, w_dim=W_DIM,
    mixing_prob=MIXING_PROB, logres=LOG_RESOLUTION, lambda_gp=LAMBDA_GP, num_gen_layers=NUM_GEN_LAYERS, loss=LOSS_FN,
    train_images_l=TRAIN_IMAGES_L, train_masks=TRAIN_MASKS, train_images_u=TRAIN_IMAGES_U, train_cross=TRAIN_CROSS,
    train_joint=TRAIN_JOINT, cross_level_train=CROSS_LEVEL_TRAIN, cross_level_test=CROSS_LEVEL_TEST,
    use_custom_nets=USE_CUSTOM_NETS, save_models=SAVE_MODELS, save_models_freq=SAVE_MODELS_FREQ,
    save_models_dir=SAVE_MODELS_DIR, save_images=SAVE_IMAGES, save_images_freq=SAVE_IMAGES_FREQ,
    save_images_dir=SAVE_IMAGES_DIR
)