from __future__ import print_function
import numpy as np
import glob
import os
import random
import gzip
import time
from gan.utils.misc import get_image, imread, REPO_ROOT


class Dataset:
    def __init__(self, name, in_height, in_width, out_height, out_width, crop=True, **kwargs):
        self.name = name

        self.in_height = in_height
        self.in_width = in_width
        self.out_height = out_height
        self.out_width = out_width
        self.crop = crop

        self.data = None  # will be loaded by self.load()

    def load(self, *args, **kwargs):
        """Load data and check the channel number `c_dim`.
        """
        raise NotImplementedError()

    def next_batch(self, batch_size):
        raise NotImplementedError()

    def _check_color(self, img):
        # read the color channel number
        self.c_dim = img.shape[-1] if len(img.shape) >= 3 else 1  # channel number (int)
        self.grayscale = (self.c_dim == 1)  # whether in grayscale (bool)


class SimpleDataset(Dataset):
    """No classes; just a bunch of images.
    """

    def load(self, image_name_pattern='*.png'):
        # Only save the image file names, as loading them all into the memory might be too much.
        self.data = glob.glob(os.path.join("data", self.name, image_name_pattern))
        self._check_color(imread(self.data[0]))

    def _next_batch_per_epoch(self, batch_size):
        """Yields next mini-batch within one epoch.
        """
        num_batches = len(self.data) // batch_size

        for batch_idx in range(num_batches):
            batch_files = self.data[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch = [get_image(
                batch_file,
                input_height=self.in_height,
                input_width=self.in_width,
                resize_height=self.out_height,
                resize_width=self.out_width,
                crop=self.crop,
                grayscale=self.grayscale
            ) for batch_file in batch_files]

            if self.grayscale:
                batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
            else:
                batch_images = np.array(batch).astype(np.float32)

            yield batch_idx, batch_images

    def next_batch(self, batch_size):
        """Loop through batches for infinite epoches.
        """
        epoch = 0
        while True:
            epoch += 1
            random.shuffle(self.data)  # shuffle in place.
            for step, train_data in self._next_batch_per_epoch(batch_size):
                yield epoch, step, train_data


def load_mnist(data_dir, y_dim=10):
    assert os.path.exists(data_dir)

    def _load_and_reshape(file_name, size, offset):
        f = gzip.GzipFile(os.path.join(data_dir, file_name))
        loaded = np.fromstring(f.read(), dtype=np.uint8)
        return loaded[offset:].reshape(size).astype(np.float)

    train_X = _load_and_reshape('train-images-idx3-ubyte.gz', (60000, 28, 28, 1), 16)
    train_y = _load_and_reshape('train-labels-idx1-ubyte.gz', (60000), 8)
    train_y = np.asarray(train_y)

    test_X = _load_and_reshape('t10k-images-idx3-ubyte.gz', (10000, 28, 28, 1), 16)
    test_y = _load_and_reshape('t10k-labels-idx1-ubyte.gz', (10000), 8)
    test_y = np.asarray(test_y)

    X = np.concatenate((train_X, test_X), axis=0)
    y = np.concatenate((train_y, test_y), axis=0).astype(np.int)

    # Shuffling X and y in the same order.
    seed = int(time.time())
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    # Simple one-hot encoding
    y_vec = np.zeros((len(y), y_dim), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    # Normalize the image features
    X /= 255.

    return X, y_vec


class MnistDataset(Dataset):
    """Dataset with y classes: 'mnist' and 'fashion-mnist'.
    """

    def __init__(self, name, in_height, in_width, out_height, out_width,
                 y_dim=10, **kwargs):
        super().__init__(name, in_height, in_width, out_height, out_width, **kwargs)

        self.y_dim = y_dim  # depends on the dataset.
        self.data_y = None  # will be loaded by self.load()

    def load(self):
        if self.name in ['mnist', 'fashion-mnist']:
            data_dir = os.path.join(REPO_ROOT, "data", self.name)
            self.data, self.data_y = load_mnist(data_dir, y_dim=self.y_dim)
        else:
            raise NotImplementedError()

        self._check_color(self.data[0])

    def get_next_batch_one_epoch(self, num_batches, config):
        """Yields next mini-batch within one epoch.
        """
        for idx in range(0, num_batches):
            batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

            batch_images = self.data_X[idx * config.batch_size:(idx + 1) * config.batch_size]
            batch_labels = self.data_y[idx * config.batch_size:(idx + 1) * config.batch_size]

            d_train_feed_dict = {self.inputs: batch_images, self.z: batch_z,
                                 self.y: batch_labels}
            g_train_feed_dict = {self.z: batch_z, self.y: batch_labels}

            yield idx, d_train_feed_dict, g_train_feed_dict
