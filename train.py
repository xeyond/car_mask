from unet_model import Unet
from utils import read_data
import tensorflow as tf

image_size = [320, 480]
unet = Unet(input_shape=image_size)
unet.build_net()
X, Y = read_data(r'E:\Kaggle\dataset\train', r'E:\Kaggle\dataset\train_masks', n_images=50, image_size=image_size)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
with tf.Session(config=config) as sess:
    unet.train(sess, X, Y, batch_size=2, learning_rate=0.0001)

