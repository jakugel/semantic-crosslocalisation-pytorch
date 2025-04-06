import torch
from generator_network import Generator
from generator_network_custom import GeneratorCustom
from discriminator_networks_semantic import DiscriminatorImage, DiscriminatorMask, DiscriminatorJoint
from discriminator_networks_semantic_custom import DiscriminatorImageCustom, DiscriminatorMaskCustom, DiscriminatorJointCustom
from mapping_networks_semantic import MappingNetwork
from mapping_networks_semantic_custom import MappingNetworkCustom
from math import log2

from synthesise_data_semantic import generate_examples

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
NUM_CLASSES = 4
IMG_SIZE = 128
LOG_RESOLUTION = int(log2(IMG_SIZE))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EQUALIZED = False    # whether to use equalized layers
CROSS_LEVEL_TEST = NUM_GEN_LAYERS - 1
USE_CUSTOM_NETS = True
SAVE_MODELS_DIR = "./crossloc_semantic_models"
MODEL_EPOCH = 2000
SAVE_IMAGES_DIR = "./crossloc_semantic_images_test"

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

critici.load_state_dict(
        torch.load(SAVE_MODELS_DIR + "/critici_" + str(MODEL_EPOCH) + ".pth"))
criticm.load_state_dict(
        torch.load(SAVE_MODELS_DIR + "/criticm_" + str(MODEL_EPOCH) + ".pth"))
criticj.load_state_dict(
        torch.load(SAVE_MODELS_DIR + "/criticj_" + str(MODEL_EPOCH) + ".pth"))
mapnetl.load_state_dict(
    torch.load(SAVE_MODELS_DIR + "/mapnetl_" + str(MODEL_EPOCH) + ".pth"))
mapnetu.load_state_dict(
    torch.load(SAVE_MODELS_DIR + "/mapnetu_" + str(MODEL_EPOCH) + ".pth"))
geni.load_state_dict(
    torch.load(SAVE_MODELS_DIR + "/geni_" + str(MODEL_EPOCH) + ".pth"))
genm.load_state_dict(
    torch.load(SAVE_MODELS_DIR + "/genm_" + str(MODEL_EPOCH) + ".pth"))

critici.eval()
criticm.eval()
criticj.eval()
mapnetl.eval()
mapnetu.eval()
geni.eval()
genm.eval()

generate_examples(geni, genm, MODEL_EPOCH, CROSS_LEVEL_TEST, LOG_RESOLUTION, NUM_GEN_LAYERS, mapnetl, mapnetu,
                  W_DIM, device=DEVICE, save_path=SAVE_IMAGES_DIR, train_images_l=TRAIN_IMAGES_L, train_masks=TRAIN_MASKS,
                      train_images_u=TRAIN_IMAGES_U, train_cross=TRAIN_CROSS, train_joint=TRAIN_JOINT,
                  use_custom_nets=USE_CUSTOM_NETS)
