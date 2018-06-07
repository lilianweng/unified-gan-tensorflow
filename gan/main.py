"""
Recommendeds configs:

--model_type=GAN --learning_rate=0.0002
--model_type=WGAN --learning_rate=0.00005 --beta1=0.9
--model_type=WGAN_GP --learning_rate=0.0001 --beta1=0.5 --beta2=0.9
"""
from __future__ import print_function
import pprint
import numpy as np
import tensorflow as tf
from gan.model.dataset import load_dataset
from gan.model.model import UnifiedDCGANModel

flags = tf.app.flags
flags.DEFINE_string("model_type", "GAN", "Type of GAN model to use. [GAN]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for Adam. [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of Adam. [0.5]")
flags.DEFINE_float("beta2", 0.9, "Momentum term of Adam. [0.9]")
flags.DEFINE_integer("max_iter", 10000, "Maximum number of training iterations. [10000]")
flags.DEFINE_integer("d_iter", 5, "Num. batches used for training D model in one iteration. [5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images. [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images. [64]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")

FLAGS = flags.FLAGS
pp = pprint.PrettyPrinter()


def main(_):
    pp.pprint(flags.FLAGS.__flags)
    dataset = load_dataset('mnist')
    model = UnifiedDCGANModel(FLAGS.model_type, dataset)

    if FLAGS.train:
        model.train(FLAGS)
    else:
        if not model.load_model()[0]:
            raise Exception("[!] Train a model first, then run test mode")


if __name__ == '__main__':
    tf.app.run()
