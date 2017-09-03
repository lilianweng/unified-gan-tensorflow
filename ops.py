"""
From: https://github.com/carpedm20/DCGAN-tensorflow/blob/master/ops.py
"""
import math
import tensorflow as tf


if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(
            x,
            decay=self.momentum,
            updates_collections=None,
            epsilon=self.epsilon,
            scale=True,
            is_training=train,
            scope=self.name
        )


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis.

    Attach `y` to the channel level of `x`.
    y.shape() is expected to be (batch_size, 1, 1, num_categories)
    """
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([
        x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    """Apply convolution computation using a kernel of size (k_h, k_w) over the image
    input_ with strides (1, d_h, d_w, 1) and SAME padding.

    For example:
        i = <input image size>, k = 5, s = 2, p = k // 2 = 2
        o = (i + 2p - k) // 2 + 1 = (i - 1) // 2 + 1

    Read more: https://arxiv.org/pdf/1603.07285.pdf
               https://github.com/vdumoulin/conv_arithmetic

    Returns a tensor of shape (
        batch_size,
        (input_image_height - 1) // 2 + 1,
        (input_image_width - 1) // 2 + 1,
        output_dim,
    ).
    """
    with tf.variable_scope(name):
        w = tf.get_variable('weights', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    """Apply transposed convolution computation using a kernel of size (k_h, k_w) over the
    image input_ with strides (1, d_h, d_w, 1) and SAME padding.

    Read more: https://github.com/vdumoulin/conv_arithmetic

    Shapes:
        (This is the k layer from ther last)
        input_.shape = (batch_size, img_h // 2^k, img_w // 2^k, gf_dim * 2^k)
        output_shape = (batch_size, img_h // 2^(k-1), img_w // 2^(k-1), gf_dim * 2^(k-1))
        w.shape = (k_h, k_w, gf_dim * 2^(k-1), gf_dim * 2^k)
        biases.shape = (gf_dim * 2^(k-1), )
    """
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('weights', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    """ReLU layer"""
    return tf.maximum(x, leak * x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    """Linear regression layer.

    input_ (batch_size, dim) x matric (dim, output_dim) + biases (output_dim, )
    Returns a tensor of shape (batch_size, output_size)
    """
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
