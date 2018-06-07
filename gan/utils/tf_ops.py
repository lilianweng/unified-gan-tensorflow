"""
From: https://github.com/carpedm20/DCGAN-tensorflow/blob/master/ops.py
"""
import math
import tensorflow as tf


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis.
    Attach `y` to the channel level of `x`.
    x.shape() - (batch_size, img_width, img_height, num_channels)
    y.shape() - (batch_size, 1, 1, num_categories)
    """
    x_shapes = x.get_shape().as_list()
    y_shapes = y.get_shape().as_list()
    return tf.concat([x, y * tf.ones(x_shapes[:3] + [y_shapes[3]])], 3)


def linear(inputs, output_size, name='linear', batch_norm=False, activation_fn=None,
           train=True, return_w=False):
    """Linear regression layer.

    input_ (batch_size, dim) x matric (dim, output_dim) + biases (output_dim, )
    Returns a tensor of shape (batch_size, output_size)
    """
    with tf.variable_scope(name):
        matrix = tf.get_variable("matrix", [inputs.get_shape()[1], output_size], tf.float32)
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(0.0))
        output = tf.matmul(inputs, matrix) + bias

        if batch_norm:
            output = tf.layers.batch_normalization(output, training=train)

        if activation_fn:
            output = activation_fn(output)

        if return_w:
            return output, matrix, bias

        return output


def conv2d_layer(inputs, output_dim, k_h=5, k_w=5, d_h=2, d_w=2,
              name="conv2d", batch_norm=True, activation_fn=tf.nn.leaky_relu,
            train=True):
    """Apply convolution computation using a kernel of size (k_h, k_w) over the image
    input_ with strides (1, d_h, d_w, 1) and SAME padding.

    For example:
        i = <input image size>, k = 5, s = 2, p = k // 2 = 2
        o = (i + 2p - k) // 2 + 1 = (i - 1) // 2 + 1

    Read more:
    - https://arxiv.org/pdf/1603.07285.pdf
    - https://github.com/vdumoulin/conv_arithmetic

    Returns: a tensor of shape (
        batch_size,
        (input_image_height - 1) // 2 + 1,
        (input_image_width - 1) // 2 + 1,
        output_dim,
    ).
    """
    c_dim = inputs.get_shape().as_list()[-1]

    with tf.variable_scope(name):
        w = tf.get_variable('weights', [k_h, k_w, c_dim, output_dim])
        b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(inputs, w, strides=[1, d_h, d_w, 1], padding='SAME')
        conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())

        if batch_norm:
            conv = tf.layers.batch_normalization(conv, training=train)

        if activation_fn:
            conv = activation_fn(conv)

        return conv


def deconv2d_layer(inputs, output_shape,
                   k_h=5, k_w=5, d_h=2, d_w=2,
                   name="deconv2d", batch_norm=True, activation_fn=tf.nn.relu,
                   return_w=False):
    """Apply transposed convolution computation using a kernel of size (k_h, k_w) over the
    image input_ with strides (1, d_h, d_w, 1) and SAME padding.

    Read more about "transposed convolution": https://github.com/vdumoulin/conv_arithmetic

    Shapes:
        (This is the k layer from the last)
        input_.shape = (batch_size, img_h // 2^k, img_w // 2^k, gf_dim * 2^k)
        output_shape = (batch_size, img_h // 2^(k-1), img_w // 2^(k-1), gf_dim * 2^(k-1))
        w.shape = (k_h, k_w, gf_dim * 2^(k-1), gf_dim * 2^k)
        biases.shape = (gf_dim * 2^(k-1), )
    """
    inputs_shape = inputs.get_shape().as_list()
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('weights', [k_h, k_w, output_shape[-1], inputs_shape[-1]])
        b = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))

        deconv = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])
        deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())

        if batch_norm:
            deconv = tf.layers.batch_normalization(deconv)

        if activation_fn:
            deconv = activation_fn(deconv)

        if return_w:
            return deconv, w, b

        return deconv
