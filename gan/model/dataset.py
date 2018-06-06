from __future__ import print_function

import glob
import gzip
import os
import random

import numpy as np

from gan.utils.misc import get_image, imread, REPO_ROOT, shuffling

BASE_DATA_DIR = 'data'


class Dataset:
    def __init__(self, name, in_height=None, in_width=None,
                 out_height=None, out_width=None, crop=True, **kwargs):
        self.name = name
        self.data_dir = os.path.join(REPO_ROOT, BASE_DATA_DIR, self.name)

        self.in_height = in_height
        self.in_width = in_width
        self.out_height = out_height
        self.out_width = out_width
        self.crop = crop

        self.data = None  # will be loaded by self.load()

    def exists(self):
        return os.path.exists(self.data_dir)

    def load(self, *args, **kwargs):
        """Load data and check the channel number `c_dim`.
        """
        raise NotImplementedError()

    def next_batch(self, batch_size):
        """Loop through batches for infinite epoches.
        """
        raise NotImplementedError()

    def _check_img_attrs(self, img):
        # read the color channel number
        self.c_dim = img.shape[-1] if len(img.shape) >= 3 else 1  # channel number (int)
        self.grayscale = (self.c_dim == 1)  # whether in grayscale (bool)
        self.in_width = img.shape[0]
        self.in_height = img.shape[1]

        if self.out_width is None:
            self.out_width = self.in_width

        if self.out_height is None:
            self.out_height = self.in_height


class SimpleDataset(Dataset):
    """No classes; just a bunch of images.
    """

    def load(self, image_name_pattern='*.png'):
        # Only save the image file names, as loading them all into the memory might be too much.
        self.data = glob.glob(os.path.join(self.data_dir, image_name_pattern))
        self._check_img_attrs(imread(self.data[0]))

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
            random.shuffle(self.data)  # shuffle in place.
            for step, batch_images in self._next_batch_per_epoch(batch_size):
                yield epoch, step, batch_images
            epoch += 1


class MnistDataset(Dataset):
    """Dataset with y classes: 'mnist' and 'fashion-mnist'.
    """

    def __init__(self, name, y_dim=10, **kwargs):
        super().__init__(name, **kwargs)

        self.y_dim = y_dim  # depends on the dataset.
        self.data_y = None  # will be loaded by self.load()

    def load(self):
        if self.name in ['mnist', 'fashion-mnist']:
            self.data, self.data_y = self._load_mnist()
        else:
            raise NotImplementedError()

        self._check_img_attrs(self.data[0])

    def _next_batch_per_epoch(self, batch_size):
        num_batches = len(self.data) // batch_size

        for idx in range(0, num_batches):
            batch_images = self.data[idx * batch_size:(idx + 1) * batch_size]
            batch_labels = self.data_y[idx * batch_size:(idx + 1) * batch_size]

            yield idx, batch_images, batch_labels

    def next_batch(self, batch_size):
        epoch = 0
        while True:
            shuffling(self.data, self.data_y)
            for step, batch_images, batch_labels in self._next_batch_per_epoch(batch_size):
                yield epoch, step, batch_images, batch_labels
            epoch += 1

    def _load_mnist(self):
        assert os.path.exists(self.data_dir)

        def _load_and_reshape(file_name, size, offset):
            f = gzip.GzipFile(os.path.join(self.data_dir, file_name))
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
        shuffling(X, y)

        # Simple one-hot encoding
        y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i, y[i]] = 1.0

        # Normalize the image features
        X /= 255.

        return X, y_vec
