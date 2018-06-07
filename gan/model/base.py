from __future__ import print_function

import os
import re

import click
import tensorflow as tf


class BaseModel:
    """
    Abstract object representing a TensorFlow model for easy save and load.
    """

    def __init__(self, model_name, saver_max_to_keep=5, checkpoint_dir='checkpoints'):
        self._saver = None
        self._saver_max_to_keep = saver_max_to_keep
        self._checkpoint_dir = checkpoint_dir
        self._writer = None
        self._model_name = model_name
        self._sess = None

    def scope_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        assert len(res) > 0
        print("Variables in scope '%s'" % scope)
        for v in res:
            print("\t" + str(v))
        return res

    def save_model(self, step=None):
        click.secho(" [*] Saving checkpoints...", fg="green")
        ckpt_file = os.path.join(self.checkpoint_dir, self.model_name)
        self.saver.save(self.sess, ckpt_file, global_step=step)

    def load_model(self):
        """Return a tuple of (
            bool: whether the model could be loaded,
            int: checkpoint step count
        )"""
        click.secho(" [*] Loading checkpoints...", fg="green")

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        print(self.checkpoint_dir, ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            step = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(ckpt_name, step)

            fname = os.path.join(self.checkpoint_dir, ckpt_name)
            print(fname)
            self.saver.restore(self.sess, fname)
            click.secho(" [*] Load SUCCESS: %s" % fname, fg="green")
            return True, step
        else:
            click.secho(" [!] Load FAILED: %s" % self.checkpoint_dir, fg="red")
            return False, None

    @property
    def checkpoint_dir(self):
        ckpt_path = os.path.join(self._checkpoint_dir, self.model_name)
        os.makedirs(ckpt_path, exist_ok=True)
        return ckpt_path

    @property
    def model_name(self):
        assert self._model_name, "Not a valid model name."
        return self._model_name

    @property
    def saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=self._saver_max_to_keep)
        return self._saver

    @property
    def writer(self):
        if self._writer is None:
            writer_path = os.path.join("logs", self.model_name)
            os.makedirs(writer_path, exist_ok=True)
            self._writer = tf.summary.FileWriter(writer_path, self.sess.graph)
        return self._writer

    @property
    def sess(self):
        if self._sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.intra_op_parallelism_threads = 2
            config.inter_op_parallelism_threads = 2
            self._sess = tf.Session(config=config)

        return self._sess
