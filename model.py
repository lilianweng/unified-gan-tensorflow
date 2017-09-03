from __future__ import division

import math
import os
import re
import time
from glob import glob

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from ops import (batch_norm, concat, conv2d, conv_cond_concat, conv_out_size_same, deconv2d, linear, lrelu)
from utils import (get_image, image_manifold_size, imread, load_mnist, save_images)


class UnifiedDCGAN(object):
    # Three model types to choose from.
    GAN = "GAN"
    WGAN = "WGAN"
    WGAN_GP = "WGAN_GP"

    def __init__(self, sess, model_type,
                 input_height=108, input_width=108, crop=True,
                 batch_size=64, sample_num=64,
                 output_height=64, output_width=64,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024,
                 d_clip_limit=0.01, d_iter=5, gp_lambda=10.,
                 l1_regularizer_scale=None,
                 dataset_name='default', input_fname_pattern='*.png',
                 checkpoint_dir=None, sample_dir=None):
        """
        Construct a model object.

        Args:
            sess (tf.Session object)
            model_type (str)

            input_height (int)
            input_width (int)
            crop (bool): If True, crop the images in the center if the output size is smaller;
                otherwise, resize.
            batch_size (int): The size of batch. Should be specified before training.
            sample_num (int): Num. images in one sample.
            output_height (int)
            output_width (int)

            y_dim (int): Dimension of dim for y. [None]
            z_dim (int): Dimension of dim for Z. [100]
            gf_dim (int): Dimension of generator filters in first conv layer. [64]
            df_dim (int): Dimension of discriminator filters in first conv layer. [64]
            gfc_dim (int): Dimension of generator units for for fully connected layer. [1024]
            dfc_dim (int): Dimension of discriminator units for fully connected layer. [1024]

            d_clip_limit (float): When training "WGAN" model, the discriminator's variables are
                clamped to the range of [-d_clip_limit, d_clip_limit] after every gradient update.
            d_iter (int): Num. batches used for training D model in one iteration
            gp_lambda (float): The penalty parameter for "WGAN_GP" model.
            l1_regularizer_scale (float): If provided, add l1 regularizer on all trainable variables.

            dataset_name (str): Other than 'mnist', other images should be from ./data/{dataset_name}
                folder.
            input_fname_pattern (str): Regex for matching the image file names.
            checkpoint_dir (str): Folder name to save the model checkpoints.
            sample_dir (str): Folder name to save the sample images.
        """
        if model_type not in (self.GAN, self.WGAN, self.WGAN_GP):
            raise ValueError("Unknown model_type: '%s'.", model_type)

        self.model_type = model_type
        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.d_clip_limit = math.fabs(d_clip_limit)
        self.d_iter = d_iter
        self.l1_regularizer_scale = l1_regularizer_scale
        self.gp_lambda = gp_lambda

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern

        self.checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.sample_dir = os.path.join(sample_dir, self.model_dir)
        if not os.path.exists(self.sample_dir):
            os.mkdir(self.sample_dir)

        self.load_dataset()
        self.build_model()

    def load_dataset(self):
        """
        Load data and check the channel number `c_dim`.
        """
        if self.dataset_name == 'mnist':
            self.data_X, self.data_y = load_mnist(self.y_dim)
            self.c_dim = self.data_X[0].shape[-1]
        else:
            self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
            imreadImg = imread(self.data[0])

            if len(imreadImg.shape) >= 3:
                # check if image is a non-grayscale image by checking channel number
                self.c_dim = imread(self.data[0]).shape[-1]
            else:
                self.c_dim = 1

        self.grayscale = (self.c_dim == 1)

    def build_model(self):
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
        else:
            self.y = None

        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')

        inputs = self.inputs

        ##############################
        # Define batch normalization layers for constructing D and G networks.
        # Batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3')

        ##############################
        # Define the model structure

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.summary.histogram("z", self.z)

        self.G = self.generator(self.z, self.y)
        self.sampler = self.sampler(self.z, self.y)
        self.D, self.D_logits = self.discriminator(inputs, self.y, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.G_sum = tf.summary.image("G", self.G, max_outputs=4)
        self.inputs_sum = tf.summary.image("inputs", self.inputs, max_outputs=4)

        ##############################
        # Define loss function

        if self.model_type == self.GAN:
            # Define the loss function for Vanilla GAN.
            self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits, labels=tf.ones_like(self.D)))
            self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
            self.d_loss = self.d_loss_real + self.d_loss_fake

            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        else:
            # Define the loss function for Wasserstein GAN.
            self.d_loss_real = tf.reduce_mean(self.D_logits)
            self.d_loss_fake = tf.reduce_mean(self.D_logits_)
            self.d_loss = tf.reduce_mean(self.D_logits_) - tf.reduce_mean(self.D_logits)

            self.g_loss = - tf.reduce_mean(self.D_logits_)

            if self.model_type == self.WGAN_GP:
                # Wasserstein GAN with gradient penalty
                epsilon = tf.random_uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
                interpolated = epsilon * inputs + (1 - epsilon) * self.G
                _, self.D_logits_intp_ = self.discriminator(interpolated, self.y, reuse=True)

                # tf.gradients returns a list of sum(dy/dx) for each x in xs.
                gradients = tf.gradients(self.D_logits_intp_, [interpolated, ], name="D_logits_intp")[0]
                grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
                grad_penalty = tf.reduce_mean(tf.square(grad_l2 - 1.0))

                self.gp_loss_sum = tf.summary.scalar("grad_penalty", grad_penalty)
                self.grad_norm_sum = tf.summary.scalar("grad_norm", tf.nn.l2_loss(gradients))

                # Add gradient penalty to the discriminator's loss function.
                self.d_loss += self.gp_lambda * grad_penalty

        # Add regularizer if needed.
        if self.l1_regularizer_scale is not None:
            self.reg = tc.layers.apply_regularization(
                tc.layers.l1_regularizer(self.l1_regularizer_scale),
                weights_list=[var for var in tf.global_variables() if 'weights' in var.name]
            )
            self.reg_summ = tf.summary.histogram("l1_regularizer", self.reg)

            self.g_loss = self.g_loss + self.reg
            self.d_loss = self.d_loss + self.reg

        # Add various tf summary variables.
        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def get_next_batch_one_epoch(self, num_batches, config):
        """Yields next mini-batch within one epoch.
        """
        for idx in xrange(0, num_batches):
            batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

            if config.dataset == 'mnist':
                batch_images = self.data_X[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch_labels = self.data_y[idx * config.batch_size:(idx + 1) * config.batch_size]

                d_train_feed_dict = {self.inputs: batch_images, self.z: batch_z, self.y: batch_labels}
                g_train_feed_dict = {self.z: batch_z, self.y: batch_labels}

            else:
                batch_files = self.data[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch = [get_image(
                    batch_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale) for batch_file in batch_files]

                if self.grayscale:
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                d_train_feed_dict = {self.inputs: batch_images, self.z: batch_z}
                g_train_feed_dict = {self.z: batch_z}

            yield idx, d_train_feed_dict, g_train_feed_dict

    def inf_get_next_batch(self, config):
        """Loop through batches for infinite epoches.
        """
        if config.dataset == 'mnist':
            num_batches = min(len(self.data_X), config.train_size) // config.batch_size
        else:
            self.data = glob(os.path.join("./data", config.dataset, self.input_fname_pattern))
            num_batches = min(len(self.data), config.train_size) // config.batch_size

        epoch = 0
        while True:
            epoch += 1
            for (step, d_train_feed_dict, g_train_feed_dict) in \
                    self.get_next_batch_one_epoch(num_batches, config):
                yield epoch, step, d_train_feed_dict, g_train_feed_dict

    def get_sample_data(self, config):
        """Set up the inputs and labels of sample images.
        Samples are created periodically during training.
        """
        if config.dataset == 'mnist':
            sample_inputs = self.data_X[0:self.sample_num]
            sample_labels = self.data_y[0:self.sample_num]
        else:
            sample_files = self.data[0:self.sample_num]
            sample = [
                get_image(sample_file,
                          input_height=self.input_height,
                          input_width=self.input_width,
                          resize_height=self.output_height,
                          resize_width=self.output_width,
                          crop=self.crop,
                          grayscale=self.grayscale) for sample_file in sample_files]
            if self.grayscale:
                sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                sample_inputs = np.array(sample).astype(np.float32)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))
        sample_feed_dict = {
            self.z: sample_z,
            self.inputs: sample_inputs,
        }

        if config.dataset == 'mnist':
            sample_feed_dict.update({self.y: sample_labels})

        return sample_feed_dict

    def train(self, config):
        """Train the model!
        """
        d_clip = None

        ##############################
        # Define the optimizers
        if self.model_type == self.GAN:
            d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                .minimize(self.d_loss, var_list=self.d_vars)
            g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                .minimize(self.g_loss, var_list=self.g_vars)

        elif self.model_type == self.WGAN:
            # Wasserstein GAN
            d_optim = tf.train.RMSPropOptimizer(config.learning_rate) \
                .minimize(self.d_loss, var_list=self.d_vars)
            g_optim = tf.train.RMSPropOptimizer(config.learning_rate) \
                .minimize(self.g_loss, var_list=self.g_vars)

            # After every gradient update on the discriminator model, clamp its weights to a
            # small fixed range, [-d_clip_limit, d_clip_limit].
            d_clip = tf.group(*[v.assign(tf.clip_by_value(
                v, -self.d_clip_limit, self.d_clip_limit)) for v in self.d_vars])

        elif self.model_type == self.WGAN_GP:
            d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1, beta2=config.beta2) \
                .minimize(self.d_loss, var_list=self.d_vars)
            g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1, beta2=config.beta2) \
                .minimize(self.g_loss, var_list=self.g_vars)

        tf.global_variables_initializer().run()

        # Merge summary
        g_sum_list = [self.z_sum, self.d__sum, self.G_sum, self.g_loss_sum, self.d_loss_fake_sum]
        d_sum_list = [self.z_sum, self.d_sum, self.inputs_sum, self.d_loss_sum, self.d_loss_real_sum]

        if self.model_type in (self.WGAN, self.WGAN_GP) and self.l1_regularizer_scale is not None:
            g_sum_list += [self.reg_summ]
            d_sum_list += [self.reg_summ]

        if self.model_type == self.WGAN_GP:
            d_sum_list += [self.gp_loss_sum, self.grad_norm_sum]

        self.g_sum = tf.summary.merge(g_sum_list)
        self.d_sum = tf.summary.merge(d_sum_list)

        self.writer = tf.summary.FileWriter(os.path.join("./logs", self.model_dir), self.sess.graph)

        # Set up the sample images
        sample_feed_dict = self.get_sample_data(config)

        # Create a sample image every `sample_every_step` steps.
        sample_every_step = int(config.max_iter // 20)

        start_time = time.time()
        could_load, checkpoint_counter = self.load()

        counter = 1  # Count how many batches we have processed.
        d_counter = 0  # Count number of batches used for training D
        g_counter = 0  # Count number of batches used for training G

        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        ##############################
        # Start training!

        inf_data_gen = self.inf_get_next_batch(config)

        for iter_count in xrange(config.max_iter):
            if self.model_type == self.GAN:
                _d_iters = 1
            else:
                # For WGAN or WGAN_GP model, we are allowed to train the D network to be very good at
                # the beginning as a warm start. Because theoretically Wasserstain distance does not
                # suffer the vanishing gradient dilemma that vanila GAN is facing.
                _d_iters = 100 if iter_count < 25 or np.mod(iter_count, 500) == 0 else self.d_iter

            # Update D network
            counter += _d_iters
            d_counter += _d_iters
            for _ in range(_d_iters):
                epoch, step, d_train_feed_dict, g_train_feed_dict = inf_data_gen.next()
                self.sess.run(d_optim, feed_dict=d_train_feed_dict)
                if d_clip is not None:
                    self.sess.run(d_clip)

            summary_str = self.sess.run(self.d_sum, feed_dict=d_train_feed_dict)
            self.writer.add_summary(summary_str, iter_count)

            # Update G network
            g_counter += 1
            _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict=g_train_feed_dict)
            self.writer.add_summary(summary_str, iter_count)

            d_err = self.d_loss.eval(d_train_feed_dict)
            g_err = self.g_loss.eval(g_train_feed_dict)

            if np.mod(iter_count, 100) == 0:
                print("Iter: %d Epoch: %d [%d/%d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % (
                    iter_count, epoch, d_counter, g_counter, time.time() - start_time, d_err, g_err))

            if np.mod(iter_count, sample_every_step) == 1:
                samples, d_loss, g_loss = self.sess.run(
                    [self.sampler, self.d_loss, self.g_loss],
                    feed_dict=sample_feed_dict
                )

                image_path = os.path.join(self.sample_dir, "train_{:02d}_{:04d}.png".format(epoch, step))
                save_images(samples, image_manifold_size(samples.shape[0]), image_path)
                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                # Save the model.
                self.save(counter)

    def discriminator(self, image, y=None, reuse=False):
        """Defines the D network structure.
        """
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            if not self.y_dim:
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

                return tf.nn.sigmoid(h4), h4
            else:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)

                h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
                h1 = tf.reshape(h1, [self.batch_size, -1])
                h1 = concat([h1, y], 1)

                h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
                h2 = concat([h2, y], 1)

                h3 = linear(h2, 1, 'd_h3_lin')

                return tf.nn.sigmoid(h3), h3

    def generator(self, z, y=None):
        """Defines the G network structure.
        """
        with tf.variable_scope("generator") as scope:
            if not self.y_dim:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                # project `z` and reshape
                self.z_, self.h0_w, self.h0_b = linear(
                    z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)

                self.h0 = tf.reshape(
                    self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(self.h0))

                self.h1, self.h1_w, self.h1_b = deconv2d(
                    h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.g_bn1(self.h1))

                h2, self.h2_w, self.h2_b = deconv2d(
                    h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
                h2 = tf.nn.relu(self.g_bn2(h2))

                h3, self.h3_w, self.h3_b = deconv2d(
                    h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
                h3 = tf.nn.relu(self.g_bn3(h3))

                h4, self.h4_w, self.h4_b = deconv2d(
                    h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

                return tf.nn.tanh(h4)

            else:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
                s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

                # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

                h0 = tf.nn.relu(
                    self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
                h0 = concat([h0, y], 1)

                h1 = tf.nn.relu(self.g_bn1(
                    linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin')))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
                                                    [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(
                    deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

    def sampler(self, z, y=None):
        """TODO: merge this with self.generator()?
        """
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            if not self.y_dim:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                # project `z` and reshape
                h0 = tf.reshape(
                    linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin'),
                    [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(h0, train=False))

                h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1')
                h1 = tf.nn.relu(self.g_bn1(h1, train=False))

                h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2')
                h2 = tf.nn.relu(self.g_bn2(h2, train=False))

                h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3')
                h3 = tf.nn.relu(self.g_bn3(h3, train=False))

                h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

                return tf.nn.tanh(h4)
            else:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
                s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

                # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

                h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
                h0 = concat([h0, y], 1)

                h1 = tf.nn.relu(self.g_bn1(
                    linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin'), train=False))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(
                    deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

    @property
    def model_dir(self):
        return "{}_{}_{}_{}_{}".format(
            self.model_type, self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, step):
        model_name = self.model_type + ".model"
        model_path = os.path.join(self.checkpoint_dir, model_name)
        self.saver.save(self.sess, model_path, global_step=step)

    def load(self):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
