from unet import Unet
from utils import read_test_data, read_car_img
import tensorflow as tf
from scipy import misc
import os
import numpy as np

# print(unet.output_map)
RESULT_DIR = './test_result'
CHECKPOINT_DIR = './checkpoints/epoch143_batch1_h640_w960.ckpt'
TEST_DIR = '/home/wangxiyang/dataset/kaggle/data/test'

image_size = [1280 // 2, 1920 // 2]
unet = Unet(input_shape=image_size)
unet.build_net()
# X, img_names = read_test_data('/home/wangxiyang/dataset/kaggle/data/small_test',
#                               n_images=128, image_size=image_size)
img_names = os.listdir(TEST_DIR)
img_names.sort()

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, CHECKPOINT_DIR)
    n_imgs = len(img_names)
    for i in range(n_imgs):
        print('%s %d/%d' % (img_names[i], i, n_imgs))
        img_mat = read_car_img(os.path.join(TEST_DIR, img_names[i]), image_size=image_size)
        res = unet.predict(sess, np.expand_dims(img_mat, axis=0))
        res = res.reshape(image_size)
        misc.imsave(os.path.join(RESULT_DIR, img_names[i]), res)
