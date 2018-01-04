from unet import Unet
from utils import read_car_img
import tensorflow as tf
from scipy import misc
import os
import numpy as np

# print(unet.output_map)
RESULT_DIR = './test_result'
CHECKPOINT_PATH = './checkpoints/epoch143_batch1_h640_w960.ckpt'
TEST_DIR = '/home/wangxiyang/dataset/kaggle/data/test'

image_size = [1280 // 2, 1920 // 2]
sess = tf.Session()
unet = Unet(input_shape=image_size, sess=sess)
unet.build_net()
unet.load_weights(CHECKPOINT_PATH)
img_names = os.listdir(TEST_DIR)
img_names.sort()

n_imgs = len(img_names)
for i in range(n_imgs):
    print('%s %d/%d' % (img_names[i], i, n_imgs))
    img_mat = read_car_img(os.path.join(TEST_DIR, img_names[i]), image_size=image_size)
    res = unet.predict(np.expand_dims(img_mat, axis=0))
    res = res.reshape(image_size)
    misc.imsave(os.path.join(RESULT_DIR, img_names[i]), res)
