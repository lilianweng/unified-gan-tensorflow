from __future__ import division, print_function
import click
import math
import os
import time

import numpy as np
import tensorflow as tf
from gan.model.base import BaseModel
from gan.model.dataset import Dataset
from gan.utils.tf_ops import conv_cond_concat, conv2d_layer, deconv2d_layer, linear
from gan.utils.misc import save_images, image_manifold_size, REPO_ROOT


class UnifiedDCGANModel(BaseModel):
    # Three model types to choose from.
    VALID_MODEL_TYPES = ['gan', 'wgan', 'wgan-gp', 'sa-gan']

    def __init__(self, model_type, dataset,
                 batch_size=64, sample_num=64,
                 z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024,
                 d_clip_limit=0.01, d_iter=5, gp_lambda=10., l2_reg_scale=None,
                 checkpoint_dir='checkpoints', sample_dir='samples'):
        """
        Construct a model object.

        Args:
        - model_type (str)
        - dataset (gan.model.dataset.Dataset)

        - batch_size (int): The size of batch. Should be specified before training.
        - sample_num (int): Num. images in one sample.
        - output_height (int)
        - output_width (int)

        - y_dim (int): Dimension of dim for y. [None]
        - z_dim (int): Dimension of dim for Z. [100]
        - gf_dim (int): Dimension of generator filters in first conv layer. [64]
        - df_dim (int): Dimension of discriminator filters in first conv layer. [64]
        - gfc_dim (int): Dimension of generator units for for fully connected layer. [1024]
        - dfc_dim (int): Dimension of discriminator units for fully connected layer. [1024]

        - d_clip_limit (float): When training "WGAN" model, the discriminator's variables are
            clamped to the range of [-d_clip_limit, d_clip_limit] after every gradient update.
        - d_iter (int): Num. batches used for training D model in one iteration
        - gp_lambda (float): The penalty parameter for "WGAN_GP" model.
        - l2_reg_scale (float): If provided, add L2-regularizer on all trainable variables.

        - dataset_name (str): Other than 'mnist', other images should be from ./data/{dataset_name}
            folder.
        - input_fname_pattern (str): Regex for matching the image file names.
        - checkpoint_dir (str): Folder name to save the model checkpoints.
        - sample_dir (str): Folder name to save the sample images.
        """
        assert model_type in self.VALID_MODEL_TYPES
        assert isinstance(dataset, Dataset) and dataset.is_loaded()

        self.model_type = model_type
        self.dataset = dataset

        self.batch_size = batch_size

        self.y_dim = getattr(self.dataset, 'y_dim', None)
        self.c_dim = self.dataset.c_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.d_clip_limit = math.fabs(d_clip_limit)
        self.d_iter = d_iter
        self.l2_reg_scale = l2_reg_scale
        self.gp_lambda = gp_lambda

        # For generating sample images.
        self.sample_num = sample_num
        self.sample_dir = os.path.join(REPO_ROOT, sample_dir, self.model_name)
        os.makedirs(self.sample_dir, exist_ok=True)

        super().__init__(checkpoint_dir=checkpoint_dir)

        self.build_model()

    @property
    def model_name(self):
        return "%s_%s_batch%d_%d" % (
            self.model_type,
            self.dataset.name,
            self.batch_size,
            int(time.time())
        )

    def build_model(self):
        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.image_dims, name='inputs')
        self.inputs_sum = tf.summary.image("inputs", self.inputs, max_outputs=4)

        self.y = None
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

        ##############################
        # Define the model structure

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.summary.histogram("z", self.z)

        self.G = self.generator(self.z, self.y)
        self.sampler = self.sampler(self.z, self.y)
        self.D, self.D_logits = self.discriminator(self.inputs, self.y, reuse=False)  # for real
        self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)  # for fake

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.G_sum = tf.summary.image("G", self.G, max_outputs=4)

        ##############################
        # Define loss function

        if self.model_type == 'gan':
            # Define the loss function for Vanilla GAN.
            self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits, labels=tf.ones_like(self.D)))
            self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
            self.d_loss = self.d_loss_real + self.d_loss_fake

            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        elif self.model_type in ('wgan', 'wgan-gp'):
            # Define the loss function for Wasserstein GAN.
            self.d_loss_real = tf.reduce_mean(self.D_logits)
            self.d_loss_fake = tf.reduce_mean(self.D_logits_)
            self.d_loss = tf.reduce_mean(self.D_logits_) - tf.reduce_mean(self.D_logits)

            self.g_loss = - tf.reduce_mean(self.D_logits_)

            if self.model_type == 'wgan-gp':
                # Wasserstein GAN with gradient penalty
                epsilon = tf.random_uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
                interpolated = epsilon * self.inputs + (1 - epsilon) * self.G
                _, self.D_logits_intp_ = self.discriminator(interpolated, self.y, reuse=True)

                # tf.gradients returns a list of sum(dy/dx) for each x in xs.
                gradients = \
                    tf.gradients(self.D_logits_intp_, [interpolated, ], name="D_logits_intp")[0]
                grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
                grad_penalty = tf.reduce_mean(tf.square(grad_l2 - 1.0))

                self.gp_loss_sum = tf.summary.scalar("grad_penalty", grad_penalty)
                self.grad_norm_sum = tf.summary.scalar("grad_norm", tf.nn.l2_loss(gradients))

                # Add gradient penalty to the discriminator's loss function.
                self.d_loss += self.gp_lambda * grad_penalty

        # Add L2-loss regularization if needed.
        if self.l2_reg_scale is not None:
            g_w_vars = [x for x in tf.global_variables('generator') if 'weights' in x.name]
            self.g_reg = tf.reduce_mean([tf.nn.l2_loss(x) for x in g_w_vars])
            self.g_reg_summ = tf.summary.histogram("g_l2_reg", self.g_reg)
            self.g_loss = tf.reduce_mean(self.g_loss + self.g_reg * self.l2_reg_scale)

            d_w_vars = [x for x in tf.global_variables('discriminator') if 'weights' in x.name]
            self.d_reg = tf.reduce_mean([tf.nn.l2_loss(x) for x in d_w_vars])
            self.d_reg_summ = tf.summary.histogram("d_l2_reg", self.d_reg)
            self.d_loss = tf.reduce_mean(self.d_loss + self.d_reg * self.l2_reg_scale)

        # Add various tf.summary variables.
        self.d_loss_real_sum = tf.summary.scalar("d_loss/real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss/fake", self.d_loss_fake)

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        trainable_vars = tf.trainable_variables()
        self.d_vars = [var for var in trainable_vars if 'd_' in var.name]
        self.g_vars = [var for var in trainable_vars if 'g_' in var.name]

        # Set up and merge the summary lists.
        g_sum_list = [self.z_sum, self.d__sum, self.G_sum, self.g_loss_sum, self.d_loss_fake_sum]
        d_sum_list = [self.z_sum, self.d_sum, self.inputs_sum, self.d_loss_sum,
                      self.d_loss_real_sum]

        if self.model_type in ('wgan', 'wgan-gp') and self.l2_reg_scale is not None:
            g_sum_list += [self.g_reg_summ]
            d_sum_list += [self.d_reg_summ]

        if self.model_type == 'wgan-gp':
            d_sum_list += [self.gp_loss_sum, self.grad_norm_sum]

        self.g_summary = tf.summary.merge(g_sum_list)
        self.d_summary = tf.summary.merge(d_sum_list)

    def discriminator(self, image, y=None, reuse=False, scope_name="discriminator"):
        """Defines the D network structure.
        """
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()

            if not self.y_dim:
                return self._discriminator(image)
            else:
                return self._discriminator_conditional(image, y)

    def _discriminator(self, image):
        assert not self.y_dim
        h0 = conv2d_layer(image, self.df_dim, batch_norm=False, name='d_conv_h0')
        h1 = conv2d_layer(h0, self.df_dim * 2, name='d_conv_h1')
        h2 = conv2d_layer(h1, self.df_dim * 4, name='d_conv_h2')
        h3 = conv2d_layer(h2, self.df_dim * 8, name='d_conv_h3')

        reshaped = tf.reshape(h3, [self.batch_size, -1])
        h4 = linear(reshaped, 1, 'd_lin_h4')

        return tf.nn.sigmoid(h4), h4

    def _discriminator_conditional(self, image, y):
        assert (self.y_dim is not None) and (self.y_dim > 0)
        assert y is not None

        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        inputs = conv_cond_concat(image, yb)

        h0 = conv2d_layer(inputs, self.c_dim + self.y_dim, batch_norm=False, name='d_conv_h0')
        h0 = conv_cond_concat(h0, yb)

        h1 = conv2d_layer(h0, self.df_dim + self.y_dim, name='d_conv_h1')
        h1 = tf.reshape(h1, [self.batch_size, -1])
        h1 = tf.concat([h1, y], 1)

        h2 = linear(h1, self.dfc_dim, name='d_lin_h2', batch_norm=True,
                    activation_fn=tf.nn.leaky_relu)
        h3 = linear(h2, 1, 'd_lin_h3')

        return tf.nn.sigmoid(h3), h3

    def generator(self, z, y=None, scope_name="generator", reuse=False):
        """Defines the G network structure.
        """
        with tf.variable_scope(scope_name, reuse=reuse) as scope:
            if reuse:
                scope.reuse_variables()

            if not self.y_dim:
                return self._generator(z)
            else:
                return self._generator_conditional(z, y)

    def _generator(self, z, train=True):
        assert not self.y_dim

        s_h, s_w = self.dataset.out_width, self.dataset.out_height
        s_h2, s_w2 = int(s_h / 2), int(s_w / 2)
        s_h4, s_w4 = int(s_h2 / 2), int(s_w2 / 2)
        s_h8, s_w8 = int(s_h4 / 2), int(s_w4 / 2)
        s_h16, s_w16 = int(s_h8 / 2), int(s_w8 / 2)

        # project `z` and reshape
        z_ = linear(z, self.gf_dim * 8 * s_h16 * s_w16, name='g_lin_h0')
        h0 = tf.reshape(z_, [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.layers.batch_normalization(h0, training=train)
        h0 = tf.nn.relu(h0)

        h1 = deconv2d_layer(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_deconv_h1')
        h2 = deconv2d_layer(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_deconv_h2')
        h3 = deconv2d_layer(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_deconv_h3')
        h4 = deconv2d_layer(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_deconv_h4')

        return tf.nn.tanh(h4)

    def _generator_conditional(self, z, y):
        assert (self.y_dim is not None) and (self.y_dim > 0)
        assert y is not None

        s_h, s_w = self.dataset.out_width, self.dataset.out_height
        s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
        s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = tf.concat([z, y], 1)

        h0 = linear(z, self.gfc_dim, name='g_lin_h0', batch_norm=True, activation_fn=tf.nn.relu)
        print('z', z.get_shape())
        print('h0', h0.get_shape())
        print('y', y.get_shape())
        h0 = tf.concat([h0, y], 1)

        h1 = linear(h0, self.gf_dim * 2 * s_h4 * s_w4, name='g_lin_h1',
                    batch_norm=True, activation_fn=tf.nn.relu)
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
        print('h1', h1.get_shape())
        print('yb', yb.get_shape())
        h1 = conv_cond_concat(h1, yb)

        h2 = deconv2d_layer(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_deconv_h2')
        h2 = conv_cond_concat(h2, yb)

        h3 = deconv2d_layer(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_deconv_h3',
                            batch_norm=False, activation_fn=None)

        return tf.nn.sigmoid(h3)

    def sampler(self, z, y=None):
        return self.generator(z, y=y, reuse=True)

    def get_sample_data(self):
        """Set up the inputs and labels of sample images.
        Samples are created periodically during training.
        """
        sample_feed_dict = {}
        if self.y_dim is None:
            sample_inputs = self.dataset.sample(self.sample_num)
        else:
            sample_inputs, sample_labels = self.dataset.sample(self.sample_num)
            sample_feed_dict.update({self.y: sample_labels})

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))
        sample_feed_dict.update({
            self.z: sample_z,
            self.inputs: sample_inputs,
        })

        return sample_feed_dict

    def get_next_batch(self):
        """Loop through batches for infinite epoches.
        """
        for batch in self.dataset.next_batch(self.batch_size):
            # The `batch` is a tuple of (epoch, step, images) or (epoch, step, images, labels)
            z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
            epoch, step, images = batch[:3]

            d_train_feed_dict = {self.inputs: images, self.z: z}
            g_train_feed_dict = {self.z: z}

            if self.y is not None:
                labels = batch[3]
                d_train_feed_dict.update({self.y: labels})
                g_train_feed_dict.update({self.y: labels})
            yield epoch, step, d_train_feed_dict, g_train_feed_dict

    def train(self, config):
        """Train the model!
        """
        d_clip = None

        ##############################
        # Define the optimizers
        if self.model_type == 'gan':
            d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                .minimize(self.d_loss, var_list=self.d_vars)
            g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                .minimize(self.g_loss, var_list=self.g_vars)

        elif self.model_type == 'wgan':
            # Wasserstein GAN
            d_optim = tf.train.RMSPropOptimizer(config.learning_rate) \
                .minimize(self.d_loss, var_list=self.d_vars)
            g_optim = tf.train.RMSPropOptimizer(config.learning_rate) \
                .minimize(self.g_loss, var_list=self.g_vars)

            # After every gradient update on the discriminator model, clamp its weights to a
            # small fixed range, [-d_clip_limit, d_clip_limit].
            d_clip = tf.group(*[v.assign(tf.clip_by_value(
                v, -self.d_clip_limit, self.d_clip_limit)) for v in self.d_vars])

        elif self.model_type == 'wgan-gp':
            d_optim = tf.train.AdamOptimizer(
                config.learning_rate, beta1=config.beta1, beta2=config.beta2) \
                .minimize(self.d_loss, var_list=self.d_vars)
            g_optim = tf.train.AdamOptimizer(
                config.learning_rate, beta1=config.beta1, beta2=config.beta2) \
                .minimize(self.g_loss, var_list=self.g_vars)

        self.sess.run(tf.global_variables_initializer())

        # Set up the sample images
        sample_feed_dict = self.get_sample_data()
        # Create a sample image every `sample_every_step` steps.
        sample_every_step = int(config.max_iter // 20)

        start_time = time.time()
        could_load, checkpoint_counter = self.load_model()

        counter = 1  # Count how many batches we have processed.
        d_counter = 0  # Count number of batches used for training D
        g_counter = 0  # Count number of batches used for training G

        if could_load:
            counter = checkpoint_counter
            click.secho(" [*] Load SUCCESS", fg='green')
        else:
            click.secho(" [!] Load failed...", fg="red")

        ##############################
        # Start training!

        train_data_generator = self.get_next_batch()

        for iter_count in range(config.max_iter):
            _d_iters = 1

            if self.model_type in ('wgan', 'wgan-gp'):
                # For 'wgan' or 'wgan-gp', we are allowed to train the D network to be very good
                # at the beginning as a warm start. Because theoretically Wasserstain distance
                # does not suffer the vanishing gradient dilemma that vanila GAN is facing.
                _d_iters = 100 if iter_count < 25 or np.mod(iter_count, 500) == 0 else self.d_iter

            # Update D network
            counter += _d_iters
            d_counter += _d_iters

            for _ in range(_d_iters):
                epoch, step, d_train_feed_dict, g_train_feed_dict = next(train_data_generator)
                print(epoch, step, d_train_feed_dict[self.inputs].shape)
                self.sess.run(d_optim, feed_dict=d_train_feed_dict)
                if d_clip is not None:
                    self.sess.run(d_clip)

            summary_str = self.sess.run(self.d_summary, feed_dict=d_train_feed_dict)
            self.writer.add_summary(summary_str, iter_count)

            # Update G network
            g_counter += 1
            _, summary_str = self.sess.run([g_optim, self.g_summary], feed_dict=g_train_feed_dict)
            self.writer.add_summary(summary_str, iter_count)

            with self.sess.as_default():
                d_err = self.d_loss.eval(d_train_feed_dict)
                g_err = self.g_loss.eval(g_train_feed_dict)

            if np.mod(iter_count, 1) == 0:
                print("Iter: %d Epoch: %d [%d/%d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % (
                    iter_count, epoch, d_counter, g_counter, time.time() - start_time, d_err,
                    g_err))

            if np.mod(iter_count, sample_every_step) == 1:
                samples, d_loss, g_loss = self.sess.run(
                    [self.sampler, self.d_loss, self.g_loss],
                    feed_dict=sample_feed_dict
                )

                image_path = os.path.join(self.sample_dir,
                                          "train_{:02d}_{:04d}.png".format(epoch, step))
                save_images(samples, image_manifold_size(samples.shape[0]), image_path)
                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                # Save the model.
                # self.save_model(step=counter)