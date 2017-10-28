import numpy as np
from scipy import misc
import os, sys
import matplotlib.pyplot as plt


def read_mask_img(filepath, image_size=(1280, 1920)):
    img = misc.imread(filepath, flatten=True).astype(np.float32)
    img_mat = misc.imresize(img, image_size)
    img_mat = img_mat / 255
    img_mat = np.expand_dims(img_mat, axis=-1)
    return img_mat


def read_car_img(filepath, image_size=(1280, 1920)):
    img = misc.imread(filepath, flatten=False).astype(np.float32)
    img_mat = misc.imresize(img, image_size)
    img_mat = img_mat / 255
    return img_mat


def read_data(car_dir, mask_dir, n_images=0, image_size=(1280, 1920)):
    print('loading car data from', car_dir)
    print('loading mask data from', mask_dir)
    car_list = os.listdir(car_dir)
    mask_list = os.listdir(mask_dir)
    car_list.sort()
    mask_list.sort()
    if n_images == 0:
        n_images = len(car_list)

    car_mat = np.zeros([n_images, image_size[0], image_size[1], 3], np.float32)
    mask_mat = np.zeros([n_images, image_size[0], image_size[1], 1], np.float32)
    for idx in range(n_images):
        car_file = car_list[idx]
        mask_file = mask_list[idx]

        car_mat[idx] = read_car_img(os.path.join(car_dir, car_file), image_size)
        mask_mat[idx] = read_mask_img(os.path.join(mask_dir, mask_file), image_size)
        sys.stdout.write('\r%2.1f%% completed...' % ((idx + 1.0) * 100 / n_images))
    print('')
    perm = np.arange(n_images)
    np.random.shuffle(perm)
    return car_mat[perm], mask_mat[perm]


if __name__ == '__main__':
    # img = read_car_img('../car-mask/test.jpg', [1280, 1920])
    img = read_mask_img('../car-mask/test.gif', [128, 192])
    plt.imshow(img)
    plt.show()
    print(img.shape)
