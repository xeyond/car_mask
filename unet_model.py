import tensorflow as tf
import sys


class Unet:
    def __init__(self, input_shape=(1280, 1920)):
        self.height, self.width = input_shape

    def get_conv2d(self, input_tensor, kernel_shape, name, activation='relu'):
        kernel_name = "%s_kernel" % name
        bias_name = "%s_bias" % name
        bias_shape = [kernel_shape[3]]
        kernels = tf.get_variable(name=kernel_name, shape=kernel_shape, initializer=tf.truncated_normal_initializer())
        bias = tf.get_variable(name=bias_name, shape=bias_shape, initializer=tf.truncated_normal_initializer())

        conv = tf.nn.conv2d(input_tensor, kernels, strides=[1, 1, 1, 1], padding='SAME')
        conv = tf.add(conv, bias, name=name)
        if activation == 'relu':
            conv = tf.nn.relu(conv)
        elif activation == 'sigmoid':
            conv = tf.nn.sigmoid(conv)
        return conv

    def build_net(self):
        self.input_holder = tf.placeholder(tf.float32, shape=[None, self.height, self.width, 3], name='input_holder')
        self.output_holder = tf.placeholder(tf.float32, shape=[None, self.height, self.width, 1], name='output_holder')

        conv1_1 = self.get_conv2d(self.input_holder, [3, 3, 3, 64], 'conv1_1')
        conv1_2 = self.get_conv2d(conv1_1, [3, 3, 64, 64], 'conv1_2')
        max_pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv2_1 = self.get_conv2d(max_pool1, [3, 3, 64, 128], 'conv2_1')
        conv2_2 = self.get_conv2d(conv2_1, [3, 3, 128, 128], 'conv2_2')
        max_pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv3_1 = self.get_conv2d(max_pool2, [3, 3, 128, 256], 'conv3_1')
        conv3_2 = self.get_conv2d(conv3_1, [3, 3, 256, 256], 'conv3_2')
        max_pool3 = tf.nn.max_pool(conv3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv4_1 = self.get_conv2d(max_pool3, [3, 3, 256, 512], 'conv4_1')
        conv4_2 = self.get_conv2d(conv4_1, [3, 3, 512, 512], 'conv4_2')
        max_pool4 = tf.nn.max_pool(conv4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv_center1 = self.get_conv2d(max_pool4, [3, 3, 512, 1024], 'conv_center1')
        conv_center2 = self.get_conv2d(conv_center1, [3, 3, 1024, 1024], 'conv_center2')

        # print([conv_center2.shape[1].value * 2, conv_center2.shape[2].value * 2])
        up_sampling4 = tf.image.resize_images(conv_center2,
                                              [conv_center2.shape[1].value * 2, conv_center2.shape[2].value * 2])
        up_sampling4 = self.get_conv2d(up_sampling4, [3, 3, 1024, 512], 'up_sampling4')
        up_sampling4 = tf.concat([conv4_2, up_sampling4], axis=-1)
        up_conv4_1 = self.get_conv2d(up_sampling4, [3, 3, 1024, 512], 'up_conv4_1')
        up_conv4_2 = self.get_conv2d(up_conv4_1, [3, 3, 512, 512], 'up_conv4_2')

        up_sampling3 = tf.image.resize_images(up_conv4_2,
                                              [up_conv4_2.shape[1].value * 2, up_conv4_2.shape[2].value * 2])
        up_sampling3 = self.get_conv2d(up_sampling3, [3, 3, 512, 256], 'up_sampling3')
        up_sampling3 = tf.concat([conv3_2, up_sampling3], axis=-1)
        up_conv3_1 = self.get_conv2d(up_sampling3, [3, 3, 512, 256], 'up_conv3_1')
        up_conv3_2 = self.get_conv2d(up_conv3_1, [3, 3, 256, 256], 'up_conv3_2')

        up_sampling2 = tf.image.resize_images(up_conv3_2,
                                              [up_conv3_2.shape[1].value * 2, up_conv3_2.shape[2].value * 2])
        up_sampling2 = self.get_conv2d(up_sampling2, [3, 3, 256, 128], 'up_sampling2')
        up_sampling2 = tf.concat([conv2_2, up_sampling2], axis=-1)
        up_conv2_1 = self.get_conv2d(up_sampling2, [3, 3, 256, 128], 'up_conv2_1')
        up_conv2_2 = self.get_conv2d(up_conv2_1, [3, 3, 128, 128], 'up_conv2_2')

        up_sampling1 = tf.image.resize_images(up_conv2_2,
                                              [up_conv2_2.shape[1].value * 2, up_conv2_2.shape[2].value * 2])
        up_sampling1 = self.get_conv2d(up_sampling1, [3, 3, 128, 64], 'up_sampling1')
        up_sampling1 = tf.concat([conv1_2, up_sampling1], axis=-1)
        up_conv1_1 = self.get_conv2d(up_sampling1, [3, 3, 128, 64], 'up_conv1_1')
        up_conv1_2 = self.get_conv2d(up_conv1_1, [3, 3, 64, 64], 'up_conv1_2')

        output_logits = self.get_conv2d(up_conv1_2, [1, 1, 64, 1], 'output_map', activation='None')
        self.output_map = tf.nn.sigmoid(output_logits)

        cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.output_holder, logits=output_logits)

        self.loss = cross_entropy_loss

    def train(self, sess, inputs, outputs, batch_size=5, epochs=100, learning_rate=0.0001):
        assert hasattr(self, 'loss')

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

        sess.run(tf.global_variables_initializer())
        n_batches = inputs.shape[0] // batch_size
        for epoch in range(epochs):
            total_loss = 0.
            for idx in range(n_batches):
                input_batch = inputs[idx * batch_size: (idx + 1) * batch_size]
                output_batch = outputs[idx * batch_size: (idx + 1) * batch_size]
                _, batch_loss, output_map = sess.run([optimizer, self.loss, self.output_map],
                                                     feed_dict={self.input_holder: input_batch,
                                                                self.output_holder: output_batch})
                sys.stdout.write("\repoch:%3d, idx: %4d, loss: %0.6f" % (epoch, idx, batch_loss))
                total_loss += batch_loss
            print("\nepoch %3d, average loss: %.6f" % (epoch, total_loss / n_batches))
