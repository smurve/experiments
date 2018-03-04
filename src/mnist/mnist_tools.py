import os
import gzip
import numpy as np
import tensorflow as tf

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
NUM_TRAIN = 60000
NUM_TEST = 10000
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT


def extract_data_as_nparrays(img_file, lbl_file, num_samples):
    """
    Extract data from imgs and labels zip files and return them as numpy arrays
    :param img_file: the full name of the zip file containing the images
    :param lbl_file: the full name of the zip file containing the labels
    :param num_samples: the number of examples to read from the files.
    :return: pair of numpy arrays containing image and label - both integers
    """
    with gzip.open(img_file) as bytestream:
        bytestream.read(16)

        buf = bytestream.read(IMAGE_SIZE * num_samples)
        imgs = np.frombuffer(buf, dtype=np.uint8)
        imgs = imgs.reshape(num_samples, IMAGE_HEIGHT, IMAGE_WIDTH, 1)

    with gzip.open(lbl_file) as bytestream:
        bytestream.read(8)

        buf = bytestream.read(num_samples)
        lbls = np.frombuffer(buf, dtype=np.uint8)

    return imgs, lbls


def dataset(data_dir, file_names, num_samples):
    """
    Create a dataset of images and labels from the given file names
    :param data_dir: the directory where the files are to be found
    :param file_names: tuple of image file name and label file name
    :param num_samples: number of samples to read from the file
    :return: a dataset containing raw integer tensors for images and one-hot-encoded labels
    """
    (img_zip, lbl_zip) = file_names
    img_path = os.path.join(data_dir, img_zip)
    lbl_path = os.path.join(data_dir, lbl_zip)
    imgs, lbls = extract_data_as_nparrays(img_path, lbl_path, num_samples)
    img_tensor = tf.constant(imgs)
    img_tensor = tf.reshape(img_tensor, [num_samples, IMAGE_SIZE])
    lbl_tensor = tf.one_hot(tf.constant(lbls), depth=10)
    return tf.data.Dataset.from_tensor_slices((img_tensor, lbl_tensor))


def datasets(data_dir):
    """
    Create a pair of datasets for train and test files located in data_dir.
    The file names are expected to be the well-known ones introduced by LeCun: USE-TYPE-TAG-ubyte.gz,
    where USE = 'train' or 't10k', TYPE='images' or 'labels' and 'TAG' = idx1 for labels and idx3 for images
    :param data_dir: the directory where the files are to be found
    :return: A pair of tf.data.Datasets containing raw integer tensors for images and one-hot-encoded labels
    """
    template = "%s-%s-ubyte.gz"

    def file_names(prefix):
        image_file = template % (prefix, "images-idx3")
        label_file = template % (prefix, "labels-idx1")
        return image_file, label_file

    train = dataset(data_dir, file_names("train"), NUM_TRAIN)
    test = dataset(data_dir, file_names("t10k"), NUM_TEST)
    return train, test
