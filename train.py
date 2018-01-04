from unet import Unet
from utils import read_data
import tensorflow as tf

image_size = [1280 // 2, 1920 // 2]
unet = Unet(input_shape=image_size)
unet.build_net()
X, Y = read_data('/home/wangxiyang/dataset/kaggle/data/train', '/home/wangxiyang/dataset/kaggle/data/train_masks',
                 n_images=100, image_size=image_size)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
with tf.Session() as sess:
    unet.train(sess, X, Y, epochs=1000, batch_size=1, learning_rate=0.0005)
