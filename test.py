from unet import Unet
from utils import read_test_data
import tensorflow as tf
from scipy import misc
# print(unet.output_map)
RESULT_DIR = './test_result'
CHECKPOINT_DIR = './checkpoints/epoch4_batch1_h640_w960.ckpt'

image_size = [1280 // 2, 1920 // 2]
unet = Unet(input_shape=image_size)
unet.build_net()
X = read_test_data('/home/wangxiyang/dataset/kaggle/data/small_test',
                      n_images=128, image_size=image_size)
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, CHECKPOINT_DIR)
    n_imgs = X.shape[0]
    for i in range(n_imgs):
        res = unet.predict(sess, X[i:i+1])
        res = res.reshape(image_size)
        misc.imsave('./test_result/test_%03d.jpg' % i, res)
