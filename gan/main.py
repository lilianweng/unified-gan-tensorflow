"""
Recommendeds configs:

TBA.
"""
from __future__ import print_function
import pprint
import tensorflow as tf

from gan.model.dataset import load_dataset
from gan.model.model import UnifiedDCGANModel

flags = tf.app.flags
flags.DEFINE_string("model_type", "gan", "Type of GAN model to use. [gan]")
flags.DEFINE_string("dataset_name", "mnist", "Name of the dataset. [mnist]")
flags.DEFINE_float("d_lr", 0.0002, "Learning rate for discriminator [0.0002]")
flags.DEFINE_float("g_lr", 0.0002, "Learning rate for generator [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of Adam. [0.5]")
flags.DEFINE_float("beta2", 0.9, "Momentum term of Adam. [0.9]")
flags.DEFINE_integer("max_iter", 10000, "Maximum number of training iterations. [10000]")
flags.DEFINE_integer("d_iter", 5, "Num. batches used for training D model in one iteration. [5]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images. [64]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")

FLAGS = flags.FLAGS
pp = pprint.PrettyPrinter()


def main(_):
    pp.pprint(FLAGS.flag_values_dict())
    dataset = load_dataset(FLAGS.dataset_name)
    print(dataset)
    dataset.load()
    model = UnifiedDCGANModel(FLAGS.model_type, dataset)

    if FLAGS.train:
        model.train(FLAGS)
    else:
        if not model.load_model()[0]:
            raise Exception("[!] Train a model first, then run test mode")


if __name__ == '__main__':
    tf.app.run()
