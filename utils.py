import numpy as np
from scipy import misc
import os, sys
import csv


def read_mask_img(filepath, image_size=(1280, 1920)):
    img = misc.imread(filepath, flatten=True).astype(np.float32)
    img_mat = misc.imresize(img, image_size)
    img_mat = img_mat / 255
    return img_mat


def read_car_img(filepath, image_size=(1280, 1920)):
    img = misc.imread(filepath, flatten=False).astype(np.float32)
    img_mat = misc.imresize(img, image_size)
    img_mat = img_mat / 255
    return img_mat


def read_data(car_dir, mask_dir, n_images=0, image_size=(1280, 1920), shuffle=False):
    print('loading car data from', car_dir)
    print('loading mask data from', mask_dir)
    car_list = os.listdir(car_dir)
    mask_list = os.listdir(mask_dir)
    car_list.sort()
    mask_list.sort()
    if n_images == 0:
        n_images = len(car_list)

    car_mat = np.zeros([n_images, image_size[0], image_size[1], 3], np.float32)
    mask_mat = np.zeros([n_images, image_size[0], image_size[1]], np.float32)
    for idx in range(n_images):
        car_file = car_list[idx]
        mask_file = mask_list[idx]

        car_mat[idx] = read_car_img(os.path.join(car_dir, car_file), image_size)
        mask_mat[idx] = read_mask_img(os.path.join(mask_dir, mask_file), image_size)
        sys.stdout.write('\r%2.1f%% completed...' % ((idx + 1.0) * 100 / n_images))
    print('')
    if shuffle:
        perm = np.arange(n_images)
        np.random.shuffle(perm)
        return car_mat[perm], mask_mat[perm]
    return car_mat, mask_mat


def read_test_data(car_dir, n_images=0, image_size=(1280, 1920)):
    print('loading car data from', car_dir)
    car_list = os.listdir(car_dir)
    car_list.sort()
    if n_images == 0:
        n_images = len(car_list)
    img_names = []
    car_mat = np.zeros([n_images, image_size[0], image_size[1], 3], np.float32)
    for idx in range(n_images):
        car_file = car_list[idx]
        car_mat[idx] = read_car_img(os.path.join(car_dir, car_file), image_size)
        sys.stdout.write('\r%2.1f%% completed...' % ((idx + 1.0) * 100 / n_images))
        img_names.append(car_file)
    print('')
    # perm = np.arange(n_images)
    # np.random.shuffle(perm)
    return car_mat, img_names


def mask2encoding(img_mat):
    mask_mat = (img_mat + 0.5).astype(int)
    mask_mat = np.transpose(mask_mat).reshape([-1])
    encoding_list = []
    pre_1 = False
    for i in range(mask_mat.shape[0]):
        if mask_mat[i] == 1:
            if not pre_1:
                encoding_list.append(i + 1)
                encoding_list.append(1)
                pre_1 = True
            else:
                encoding_list[-1] += 1
        else:
            pre_1 = False
    return encoding_list


def process_dir(src_dir, output_csv_path):
    csv_writer = csv.DictWriter(open(output_csv_path, 'w', newline=''), fieldnames=['img', 'rle_mask'])
    csv_writer.writeheader()
    files = os.listdir(src_dir)
    files.sort()
    for img_name in files:
        print(img_name)
        img_mat = read_mask_img(os.path.join(src_dir, img_name), (1280, 1918))
        encoding_list = mask2encoding(img_mat)
        encoding_str = ''
        for num in encoding_list:
            encoding_str += '%s ' % num
        csv_writer.writerow({'img': img_name, 'rle_mask': encoding_str})


if __name__ == '__main__':
    process_dir('./test_result', './test.csv')
    # img = read_car_img('../car-mask/test.jpg', [1280, 1920])
    # img = read_mask_img('../car-mask/test.gif', [128, 192])
    # plt.imshow(img)
    # plt.show()
    # print(img.shape)
