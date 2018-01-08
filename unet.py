import tensorflow as tf
import sys
from scipy import misc
import numpy as np


class Unet():
    def __init__(self, input_shape=(1280, 1920), sess=None, filter_num=64):
        self.height, self.width = input_shape
        self.sess = sess
        self.filter_num = filter_num
        self.is_restore = False

    def build_net(self, is_train=True):
        self.input_holder = tf.placeholder(tf.float32, shape=[None, self.height, self.width, 3], name='input_holder')
        self.output_holder = tf.placeholder(tf.float32, shape=[None, self.height, self.width], name='output_holder')
        self.is_train = tf.placeholder(tf.bool, name='is_train')
        filter_num = self.filter_num
        # is_train = self.is_train
        # is_train = False
        # batch_norm: change the momentum from 0.99 to 0.9 and it works!!! but it really waste GPUs

        # left layers
        conv1_1 = tf.layers.conv2d(self.input_holder, filters=filter_num, kernel_size=(3, 3), activation=tf.nn.relu,
                                   padding='same')
        conv1_1 = tf.layers.batch_normalization(conv1_1, training=is_train, momentum=0.9)
        conv1_2 = tf.layers.conv2d(conv1_1, filters=filter_num, kernel_size=(3, 3), activation=tf.nn.relu,
                                   padding='same')
        conv1_2 = tf.layers.batch_normalization(conv1_2, training=is_train, momentum=0.9)
        max_pooling1 = tf.layers.max_pooling2d(conv1_2, pool_size=(2, 2), strides=(2, 2))

        conv2_1 = tf.layers.conv2d(max_pooling1, filters=filter_num * 2, kernel_size=(3, 3), activation=tf.nn.relu,
                                   padding='same')
        conv2_1 = tf.layers.batch_normalization(conv2_1, training=is_train, momentum=0.9)
        conv2_2 = tf.layers.conv2d(conv2_1, filters=filter_num * 2, kernel_size=(3, 3), activation=tf.nn.relu,
                                   padding='same')
        conv2_2 = tf.layers.batch_normalization(conv2_2, training=is_train, momentum=0.9)
        max_pooling2 = tf.layers.max_pooling2d(conv2_2, pool_size=(2, 2), strides=(2, 2))

        conv3_1 = tf.layers.conv2d(max_pooling2, filters=filter_num * 4, kernel_size=(3, 3), activation=tf.nn.relu,
                                   padding='same')
        conv3_1 = tf.layers.batch_normalization(conv3_1, training=is_train, momentum=0.9)
        conv3_2 = tf.layers.conv2d(conv3_1, filters=filter_num * 4, kernel_size=(3, 3), activation=tf.nn.relu,
                                   padding='same')
        conv3_2 = tf.layers.batch_normalization(conv3_2, training=is_train, momentum=0.9)
        max_pooling3 = tf.layers.max_pooling2d(conv3_2, pool_size=(2, 2), strides=(2, 2))

        conv4_1 = tf.layers.conv2d(max_pooling3, filters=filter_num * 8, kernel_size=(3, 3), activation=tf.nn.relu,
                                   padding='same')
        conv4_1 = tf.layers.batch_normalization(conv4_1, training=is_train, momentum=0.9)
        conv4_2 = tf.layers.conv2d(conv4_1, filters=filter_num * 8, kernel_size=(3, 3), activation=tf.nn.relu,
                                   padding='same')
        conv4_2 = tf.layers.batch_normalization(conv4_2, training=is_train, momentum=0.9)
        max_pooling4 = tf.layers.max_pooling2d(conv4_2, pool_size=(2, 2), strides=(2, 2))

        # center layers
        center_layer = tf.layers.conv2d(max_pooling4, filters=filter_num * 16, kernel_size=(3, 3),
                                        activation=tf.nn.relu,
                                        padding='same')
        center_layer = tf.layers.batch_normalization(center_layer, training=is_train, momentum=0.9)
        center_layer = tf.layers.conv2d(center_layer, filters=filter_num * 16, kernel_size=(3, 3),
                                        activation=tf.nn.relu,
                                        padding='same')
        center_layer = tf.layers.batch_normalization(center_layer, training=is_train, momentum=0.9)

        # right layers
        up_conv4 = tf.layers.conv2d_transpose(center_layer, filters=filter_num * 8, kernel_size=(3, 3), strides=(2, 2),
                                              activation=tf.nn.relu,
                                              padding='same')
        up_conv4 = tf.layers.batch_normalization(up_conv4, training=is_train, momentum=0.9)
        up_conv4 = tf.concat([conv4_2, up_conv4], axis=-1)
        up_conv4_1 = tf.layers.conv2d(up_conv4, filters=filter_num * 8, kernel_size=(3, 3),
                                      activation=tf.nn.relu,
                                      padding='same')
        up_conv4_1 = tf.layers.batch_normalization(up_conv4_1, training=is_train, momentum=0.9)
        up_conv4_2 = tf.layers.conv2d(up_conv4_1, filters=filter_num * 8, kernel_size=(3, 3),
                                      activation=tf.nn.relu,
                                      padding='same')
        up_conv4_2 = tf.layers.batch_normalization(up_conv4_2, training=is_train, momentum=0.9)
        up_conv3 = tf.layers.conv2d_transpose(up_conv4_2, filters=filter_num * 4, kernel_size=(3, 3), strides=(2, 2),
                                              activation=tf.nn.relu,
                                              padding='same')
        up_conv3 = tf.layers.batch_normalization(up_conv3, training=is_train, momentum=0.9)
        up_conv3 = tf.concat([conv3_2, up_conv3], axis=-1)
        up_conv3_1 = tf.layers.conv2d(up_conv3, filters=filter_num * 4, kernel_size=(3, 3),
                                      activation=tf.nn.relu,
                                      padding='same')
        up_conv3_1 = tf.layers.batch_normalization(up_conv3_1, training=is_train, momentum=0.9)
        up_conv3_2 = tf.layers.conv2d(up_conv3_1, filters=filter_num * 4, kernel_size=(3, 3),
                                      activation=tf.nn.relu,
                                      padding='same')
        up_conv3_2 = tf.layers.batch_normalization(up_conv3_2, training=is_train, momentum=0.9)

        up_conv2 = tf.layers.conv2d_transpose(up_conv3_2, filters=filter_num * 2, kernel_size=(3, 3), strides=(2, 2),
                                              activation=tf.nn.relu,
                                              padding='same')
        up_conv2 = tf.layers.batch_normalization(up_conv2, training=is_train, momentum=0.9)
        up_conv2 = tf.concat([conv2_2, up_conv2], axis=-1)

        up_conv2_1 = tf.layers.conv2d(up_conv2, filters=filter_num * 2, kernel_size=(3, 3),
                                      activation=tf.nn.relu,
                                      padding='same')
        up_conv2_1 = tf.layers.batch_normalization(up_conv2_1, training=is_train, momentum=0.9)

        up_conv2_2 = tf.layers.conv2d(up_conv2_1, filters=filter_num * 2, kernel_size=(3, 3),
                                      activation=tf.nn.relu,
                                      padding='same')
        up_conv2_2 = tf.layers.batch_normalization(up_conv2_2, training=is_train, momentum=0.9)

        up_conv1 = tf.layers.conv2d_transpose(up_conv2_2, filters=filter_num, kernel_size=(3, 3), strides=(2, 2),
                                              activation=tf.nn.relu,
                                              padding='same')
        up_conv1 = tf.layers.batch_normalization(up_conv1, training=is_train, momentum=0.9)
        up_conv1 = tf.concat([conv1_2, up_conv1], axis=-1)
        up_conv1_1 = tf.layers.conv2d(up_conv1, filters=filter_num, kernel_size=(3, 3),
                                      activation=tf.nn.relu,
                                      padding='same')
        up_conv1_1 = tf.layers.batch_normalization(up_conv1_1, training=is_train, momentum=0.9)
        up_conv1_2 = tf.layers.conv2d(up_conv1_1, filters=filter_num, kernel_size=(3, 3),
                                      activation=tf.nn.relu,
                                      padding='same')
        up_conv1_2 = tf.layers.batch_normalization(up_conv1_2, training=is_train, momentum=0.9)

        # output layer
        mask_layer_logits = tf.layers.conv2d(up_conv1_2, filters=1, kernel_size=(1, 1), activation=None,
                                             padding='same')
        mask_layer_logits = tf.squeeze(mask_layer_logits, axis=-1)
        mask_layer = tf.nn.sigmoid(mask_layer_logits)
        self.output_mask = mask_layer
        print(mask_layer)
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=mask_layer_logits, labels=self.output_holder))
        print(self.loss)

        self.dice_coef = self.dice_coef(mask_layer, self.output_holder)
        print(self.dice_coef)
        self.saver = tf.train.Saver(max_to_keep=10)

    def dice_coef(self, output_map, mask):
        map_mask = tf.layers.flatten(output_map)
        mask = tf.layers.flatten(mask)
        map_mask = tf.cast(tf.greater(map_mask, 0.5), tf.float32)
        dice_numerator = 2 * tf.reduce_sum(mask * map_mask, axis=1)
        dice_denominator = tf.reduce_sum(mask, axis=1) + tf.reduce_sum(map_mask, axis=1)
        return dice_numerator / dice_denominator

    def predict(self, inputs):
        assert self.sess
        return self.sess.run(self.output_mask, feed_dict={self.input_holder: inputs, self.is_train: False})

    def predict_test(self, inputs, masks):
        assert self.sess
        return self.sess.run([self.output_mask, self.dice_coef],
                             feed_dict={self.input_holder: inputs, self.output_holder: masks, self.is_train: False})

    def load_weights(self, checkpoint_path):
        assert self.sess
        assert hasattr(self, 'saver')
        self.saver.restore(self.sess, checkpoint_path)
        self.is_restore = True

    def save_weights(self, checkpoint_path):
        assert self.sess
        assert hasattr(self, 'saver')
        self.saver.save(self.sess, checkpoint_path)

    def train(self, images, masks, val_images, val_masks, batch_size=1, epochs=100, learning_rate=0.001):
        assert hasattr(self, 'loss')
        assert self.sess is not None

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
            optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(self.loss)
        # optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(self.loss)
        if not self.is_restore:
            self.sess.run(tf.global_variables_initializer())
        n_batches = images.shape[0] // batch_size
        t_batches = val_images.shape[0]
        val_min_loss = 100.
        val_max_dice = 0.
        for epoch in range(epochs):
            # train
            total_loss = 0.
            total_dice = 0.
            for idx in range(n_batches):
                input_batch = images[idx * batch_size: (idx + 1) * batch_size]
                output_batch = masks[idx * batch_size: (idx + 1) * batch_size]
                _, batch_loss, output_map, dice_coef = self.sess.run(
                    [optimizer, self.loss, self.output_mask, self.dice_coef],
                    feed_dict={self.input_holder: input_batch,
                               self.output_holder: output_batch,
                               self.is_train: True})
                sys.stdout.write("\r-train epoch:%3d, idx: %4d, loss: %0.6f, dice_coef: %.6f" % (
                    epoch, idx, batch_loss, np.mean(dice_coef)))
                total_loss += batch_loss
                total_dice += np.mean(dice_coef)
                # misc.imsave('./test_result/train_%03d_epoch%03d.jpg' % (idx, epoch),output_map.reshape([self.height, self.width]))
            print("\n-train epoch:%3d, average loss: %.6f, average dice: %.6f" % (
                epoch, total_loss / n_batches, total_dice / n_batches))

            # validation
            total_loss = 0.
            total_dice = 0
            t_batch_size = 1
            for idx in range(t_batches):
                input_batch = val_images[idx * t_batch_size: (idx + 1) * t_batch_size]
                output_batch = val_masks[idx * t_batch_size: (idx + 1) * t_batch_size]
                batch_loss, output_map, dice_coef = self.sess.run(
                    [self.loss, self.output_mask, self.dice_coef],
                    feed_dict={self.input_holder: input_batch,
                               self.output_holder: output_batch,
                               self.is_train: False})
                sys.stdout.write("\r--test epoch:%3d, idx: %4d, loss: %0.6f, dice_coef: %.6f" % (
                    epoch, idx, batch_loss, np.mean(dice_coef)))
                total_loss += batch_loss
                total_dice += np.mean(dice_coef)
            avg_loss = total_loss / t_batches
            avg_dice = total_dice / t_batches
            print("\n--test epoch:%3d, average loss: %.6f, average dice: %.6f" % (
                epoch, avg_loss, avg_dice))
            model_need_save = False
            if val_min_loss > avg_loss:
                val_min_loss = avg_loss
                model_need_save = True
            if val_max_dice < avg_dice:
                val_max_dice = avg_dice
                model_need_save = True
            always_save = True
            if model_need_save or always_save:
                checkpoint_path = './checkpoints/epoch%d_batch%d_h%d_w%d_filter%d_loss%04d_dice_%04d_bn.ckpt' % (
                    epoch, batch_size, self.height, self.width, self.filter_num, int(avg_loss * 10000),
                    int(avg_dice * 10000))
                self.save_weights(checkpoint_path)
                print('saved weights:', checkpoint_path)
