from unet_model import Unet
from utils import read_data
import tensorflow as tf

unet = Unet()
unet.build_net()
X, Y = read_data(r'E:\Kaggle\dataset\train', r'E:\Kaggle\dataset\train_masks', n_images=100)

with tf.Session() as sess:
    unet.train(sess, X, Y, batch_size=2, learning_rate=0.0001)

