"""
    tf.data.Dataset interface to the MNIST dataset.
    Based on:
    "Convolutional Neural Network Estimator for MNIST, built with tf.layers"
    https://github.com/tensorflow/models/tree/master/official/mnist
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import gzip
import numpy as np
from six.moves import urllib
import tensorflow as tf


def train_input_fn(data_dir, batch_size, epochs):
    def _input_fn():
        ds = train_dataset(data_dir)
        ds = ds.cache().shuffle(buffer_size=50000).batch(batch_size).repeat(epochs)
        return ds
    return _input_fn


def eval_input_fn(data_dir, batch_size):
    def _input_fn():
        return test_dataset(data_dir).batch(
            batch_size).make_one_shot_iterator().get_next()
    return _input_fn


def read32(bytestream):
    """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def check_image_file_header(filename):
    """Validate that filename corresponds to images for the MNIST dataset."""
    with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        read32(f)
        rows = read32(f)
        cols = read32(f)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                           f.name))
        if rows != 28 or cols != 28:
            raise ValueError(
                'Invalid MNIST file %s: Expected 28x28 images, found %dx%d' %
                (f.name, rows, cols))


def check_labels_file_header(filename):
    """Validate that filename corresponds to labels for the MNIST dataset."""
    with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        read32(f)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                           f.name))


def download(directory, filename):
    """Download (and unzip) a file from the MNIST dataset if not already done."""
    filepath = os.path.join(directory, filename)
    if tf.gfile.Exists(filepath):
        return filepath
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + filename + '.gz'
    zipped_filepath = filepath + '.gz'
    print('Downloading %s to %s' % (url, zipped_filepath))
    # noinspection PyUnresolvedReferences
    urllib.request.urlretrieve(url, zipped_filepath)
    with gzip.open(zipped_filepath, 'rb') as f_in, open(filepath, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(zipped_filepath)
    return filepath


def dataset(directory, images_file, labels_file):
    images_file = download(directory, images_file)
    labels_file = download(directory, labels_file)

    check_image_file_header(images_file)
    check_labels_file_header(labels_file)

    def decode_image(image):
        # Normalize from [0, 255] to [0.0, 1.0]
        image = tf.decode_raw(image, tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [784])
        return image / 255.0

    def decode_label(label):
        label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
        label = tf.reshape(label, [])  # label is a scalar
        return tf.to_int32(label)

    images = tf.data.FixedLengthRecordDataset(
        images_file, 28 * 28, header_bytes=16).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
        labels_file, 1, header_bytes=8).map(decode_label)
    return tf.data.Dataset.zip((images, labels))


def train_dataset(directory):
    """tf.data.Dataset object for MNIST training data."""
    return dataset(directory, 'train-images-idx3-ubyte',
                   'train-labels-idx1-ubyte')


def test_dataset(directory):
    """tf.data.Dataset object for MNIST test data."""
    return dataset(directory, 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')
