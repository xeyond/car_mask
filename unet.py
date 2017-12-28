import tensorflow as tf
import sys
from scipy import misc

class Unet():
    def __init__(self, input_shape=(1280, 1920)):
        self.height, self.width = input_shape

    def build_net(self):
        self.input_holder = tf.placeholder(tf.float32, shape=[None, self.height, self.width, 3], name='input_holder')
        self.output_holder = tf.placeholder(tf.float32, shape=[None, self.height, self.width], name='output_holder')

        filter_num = 44
        conv1_1 = tf.layers.conv2d(self.input_holder, filters=filter_num, kernel_size=(3, 3), activation=tf.nn.relu,
                                   padding='same')
        conv1_2 = tf.layers.conv2d(conv1_1, filters=filter_num, kernel_size=(3, 3), activation=tf.nn.relu,
                                   padding='same')
        max_pooling1 = tf.layers.max_pooling2d(conv1_2, pool_size=(2, 2), strides=(2, 2))

        conv2_1 = tf.layers.conv2d(max_pooling1, filters=filter_num * 2, kernel_size=(3, 3), activation=tf.nn.relu,
                                   padding='same')
        conv2_2 = tf.layers.conv2d(conv2_1, filters=filter_num * 2, kernel_size=(3, 3), activation=tf.nn.relu,
                                   padding='same')
        max_pooling2 = tf.layers.max_pooling2d(conv2_2, pool_size=(2, 2), strides=(2, 2))

        conv3_1 = tf.layers.conv2d(max_pooling2, filters=filter_num * 4, kernel_size=(3, 3), activation=tf.nn.relu,
                                   padding='same')
        conv3_2 = tf.layers.conv2d(conv3_1, filters=filter_num * 4, kernel_size=(3, 3), activation=tf.nn.relu,
                                   padding='same')
        max_pooling3 = tf.layers.max_pooling2d(conv3_2, pool_size=(2, 2), strides=(2, 2))

        conv4_1 = tf.layers.conv2d(max_pooling3, filters=filter_num * 8, kernel_size=(3, 3), activation=tf.nn.relu,
                                   padding='same')
        conv4_2 = tf.layers.conv2d(conv4_1, filters=filter_num * 8, kernel_size=(3, 3), activation=tf.nn.relu,
                                   padding='same')
        max_pooling4 = tf.layers.max_pooling2d(conv4_2, pool_size=(2, 2), strides=(2, 2))

        center_layer = tf.layers.conv2d(max_pooling4, filters=filter_num * 16, kernel_size=(3, 3),
                                        activation=tf.nn.relu,
                                        padding='same')
        center_layer = tf.layers.conv2d(center_layer, filters=filter_num * 16, kernel_size=(3, 3),
                                        activation=tf.nn.relu,
                                        padding='same')
        up_sampling4 = tf.keras.layers.UpSampling2D((2, 2))(center_layer)
        up_sampling4 = tf.layers.conv2d(up_sampling4, filters=filter_num * 8, kernel_size=(3, 3),
                                        activation=tf.nn.relu,
                                        padding='same')
        up_sampling4 = tf.concat([conv4_2, up_sampling4], axis=-1)
        up_conv4_1 = tf.layers.conv2d(up_sampling4, filters=filter_num * 8, kernel_size=(3, 3),
                                      activation=tf.nn.relu,
                                      padding='same')
        up_conv4_2 = tf.layers.conv2d(up_conv4_1, filters=filter_num * 8, kernel_size=(3, 3),
                                      activation=tf.nn.relu,
                                      padding='same')

        up_sampling3 = tf.keras.layers.UpSampling2D((2, 2))(up_conv4_2)
        up_sampling3 = tf.layers.conv2d(up_sampling3, filters=filter_num * 4, kernel_size=(3, 3),
                                        activation=tf.nn.relu,
                                        padding='same')
        up_sampling3 = tf.concat([conv3_2, up_sampling3], axis=-1)
        up_conv3_1 = tf.layers.conv2d(up_sampling3, filters=filter_num * 4, kernel_size=(3, 3),
                                      activation=tf.nn.relu,
                                      padding='same')
        up_conv3_2 = tf.layers.conv2d(up_conv3_1, filters=filter_num * 4, kernel_size=(3, 3),
                                      activation=tf.nn.relu,
                                      padding='same')

        up_sampling2 = tf.keras.layers.UpSampling2D((2, 2))(up_conv3_2)
        up_sampling2 = tf.layers.conv2d(up_sampling2, filters=filter_num * 2, kernel_size=(3, 3),
                                        activation=tf.nn.relu,
                                        padding='same')
        up_sampling2 = tf.concat([conv2_2, up_sampling2], axis=-1)
        up_conv2_1 = tf.layers.conv2d(up_sampling2, filters=filter_num * 2, kernel_size=(3, 3),
                                      activation=tf.nn.relu,
                                      padding='same')
        up_conv2_2 = tf.layers.conv2d(up_conv2_1, filters=filter_num * 2, kernel_size=(3, 3),
                                      activation=tf.nn.relu,
                                      padding='same')

        up_sampling1 = tf.keras.layers.UpSampling2D((2, 2))(up_conv2_2)
        up_sampling1 = tf.layers.conv2d(up_sampling1, filters=filter_num, kernel_size=(3, 3),
                                        activation=tf.nn.relu,
                                        padding='same')
        up_sampling1 = tf.concat([conv1_2, up_sampling1], axis=-1)
        up_conv1_1 = tf.layers.conv2d(up_sampling1, filters=filter_num, kernel_size=(3, 3),
                                      activation=tf.nn.relu,
                                      padding='same')
        up_conv1_2 = tf.layers.conv2d(up_conv1_1, filters=filter_num, kernel_size=(3, 3),
                                      activation=tf.nn.relu,
                                      padding='same')
        # print(up_conv1_2)

        mask_layer = tf.layers.conv2d(up_conv1_2, filters=1, kernel_size=(1, 1), activation=tf.nn.sigmoid,
                                      padding='same')
        mask_layer = tf.squeeze(mask_layer, axis=-1)
        self.output_mask = mask_layer
        print(mask_layer)
        self.loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(mask_layer, self.output_holder))
        print(self.loss)


    def predict(self, sess, inputs):
        return sess.run(self.output_mask, feed_dict={self.input_holder:inputs})
    

    def train(self, sess, inputs, outputs, batch_size=1, epochs=100, learning_rate=0.001):
        assert hasattr(self, 'loss')
        saver = tf.train.Saver()
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)
        sess.run(tf.global_variables_initializer())
        n_batches = inputs.shape[0] // batch_size
        for epoch in range(epochs):
            total_loss = 0.
            for idx in range(n_batches):
                input_batch = inputs[idx * batch_size: (idx + 1) * batch_size]
                output_batch = outputs[idx * batch_size: (idx + 1) * batch_size]
                _, batch_loss, output_map = sess.run([optimizer, self.loss, self.output_mask],
                                                     feed_dict={self.input_holder: input_batch,
                                                                self.output_holder: output_batch})
                sys.stdout.write("\repoch:%3d, idx: %4d, loss: %0.6f" % (epoch, idx, batch_loss))
                total_loss += batch_loss
                #misc.imsave('./test_result/train_%03d_epoch%03d.jpg' % (idx, epoch),output_map.reshape([self.height, self.width]))
            print("\nepoch %3d, average loss: %.6f" % (epoch, total_loss / n_batches))
            if (epoch + 1) % 1 == 0:
                saver.save(sess,
                           './checkpoints/epoch%d_batch%d_h%d_w%d.ckpt' % (epoch, batch_size, self.height, self.width))
