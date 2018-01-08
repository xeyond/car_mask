from unet import Unet
from utils import read_car_img, read_mask_img
import tensorflow as tf
from scipy import misc
import os
import numpy as np
import argparse


def build_parser():
    parser = argparse.ArgumentParser()

    # model parameters
    parser.add_argument('--img_width', type=int, default=960)
    parser.add_argument('--img_height', type=int, default=640)
    parser.add_argument('--filter_num', type=int, default=44)
    parser.add_argument('--batch_norm', action='store_true', default=False)

    # checkpoint path
    parser.add_argument('--checkpoint_path', type=str, default=None)

    # test data and result dir
    parser.add_argument('--test_dir', type=str, default='/home/wangxiyang/dataset/kaggle/data/small_test')
    parser.add_argument('--mask_dir', type=str, default=None)
    parser.add_argument('--result_dir', type=str, default=None)
    parser.add_argument('--n_images', type=int, default=0)

    return parser


def main():
    args = build_parser().parse_args()
    assert args.checkpoint_path

    result_dir = args.result_dir
    checkpoint_path = args.checkpoint_path
    test_dir = args.test_dir
    n_imgs = args.n_images

    image_size = [args.img_height, args.img_width]
    sess = tf.Session()
    unet = Unet(input_shape=image_size, sess=sess, filter_num=args.filter_num, batch_norm=args.batch_norm)
    unet.build_net(is_train=False)
    unet.load_weights(checkpoint_path)
    img_names = os.listdir(test_dir)
    img_names.sort()
    mask_names = None
    total_dice = None
    if args.mask_dir:
        mask_names = os.listdir(args.mask_dir)
        mask_names.sort()
        total_dice = 0

    if n_imgs <= 0:
        n_imgs = len(img_names)

    for i in range(n_imgs):
        print('%s %d/%d' % (img_names[i], i, n_imgs))
        img_mat = read_car_img(os.path.join(test_dir, img_names[i]), image_size=image_size)
        img_mat = np.expand_dims(img_mat, axis=0)
        if mask_names:
            mask_mat = read_mask_img(os.path.join(args.mask_dir, mask_names[i]), image_size=image_size)
            mask_mat = np.expand_dims(mask_mat, axis=0)
            res, dice = unet.predict_test(img_mat, mask_mat)
            dice = np.mean(dice)
            print('Dice coefficient:%.6f' % dice)
            total_dice += dice
        else:
            res = unet.predict(np.expand_dims(img_mat, axis=0))

        if args.result_dir:
            res = res.reshape(image_size)
            misc.imsave(os.path.join(result_dir, img_names[i]), res)
    if total_dice:
        print('Average Dice coefficient:%.6f' % (total_dice / n_imgs))


if __name__ == '__main__':
    main()
