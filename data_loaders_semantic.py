import h5py
import numpy as np


def get_data_labelled(h5filepath):
    h5_file = h5py.File(h5filepath, "r")

    images_labelled = np.transpose(h5_file["images"][:], axes=(0, 3, 2, 1))
    masks = np.transpose(h5_file["labels"][:], axes=(0, 3, 2, 1))

    return images_labelled, masks


def get_data_unlabelled(h5filepath):
    h5_file = h5py.File(h5filepath, "r")

    images_unlabelled = np.transpose(h5_file["images"][:], axes=(0, 3, 2, 1))

    return images_unlabelled