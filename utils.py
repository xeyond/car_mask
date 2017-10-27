import numpy as np
from scipy import misc
import os
import matplotlib.pyplot as plt


def read_mask_img(filepath):
    img = misc.imread(filepath, flatten=True).astype(np.float32)
    img_mat = np.zeros([1280, 1920], np.float32)
    img_mat[:, 1:1919] = img
    img_mat[:, 1919] = img[:, 1917]
    img_mat[:, 0] = img[:, 0]
    img_mat = np.expand_dims(img_mat, axis=-1)
    # print(img_mat.shape, img_mat.dtype)
    return img_mat


def read_car_img(filepath):
    img = misc.imread(filepath, flatten=False).astype(np.float32)
    img_mat = np.zeros([1280, 1920, 3], np.float32)
    img_mat[:, 1:1919, :] = img
    img_mat[:, 1919, :] = img[:, 1917, :]
    img_mat[:, 0, :] = img[:, 0, :]
    return img_mat


def read_data(car_dir, mask_dir, n_images=0):
    car_list = os.listdir(car_dir)
    mask_list = os.listdir(mask_dir)
    car_list.sort()
    mask_list.sort()
    if n_images == 0:
        n_images = len(car_list)

    car_mat = np.zeros([n_images, 1280, 1920, 3], np.float32)
    mask_mat = np.zeros([n_images, 1280, 1920, 1], np.float32)
    for idx in range(n_images):
        car_file = car_list[idx]
        mask_file = mask_list[idx]
        print(idx, car_file, mask_file)
        car_mat[idx] = read_car_img(os.path.join(car_dir, car_file))
        mask_mat[idx] = read_mask_img(os.path.join(mask_dir, mask_file))

    perm = np.arange(n_images)
    np.random.shuffle(perm)
    return car_mat[perm], mask_mat[perm]
