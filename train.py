from unet_model import Unet
from utils import read_data
import tensorflow as tf

unet = Unet()
unet.build_net()
X, Y = read_data(r'E:\Kaggle\dataset\train', r'E:\Kaggle\dataset\train_masks', n_images=100)
sess = tf.Session()
unet.train(sess, X, Y, 2)
sess.close()
# print(unet.output_map)

