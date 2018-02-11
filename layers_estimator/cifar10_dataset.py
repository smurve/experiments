import tensorflow as tf
import os

# filename = "/var/ellie/data/cifar10_tfr/eval.tfrecords"
DEPTH = 3
HEIGHT = 32
WIDTH = 32


def parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([DEPTH * HEIGHT * WIDTH])

    # Reshape from [depth * height * width] to [depth, height, width].
    image = tf.cast(
        tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
        tf.float32)
    label = tf.cast(features['label'], tf.int32)

    return image, label


def dataset(filename):
    res = tf.data.TFRecordDataset(filename).map(parser)
    return res


def train_dataset(directory):
    """tf.data.Dataset object for MNIST training data."""
    return dataset(os.path.join(directory, 'train.tfrecords'))


def test_dataset(directory):
    """tf.data.Dataset object for MNIST test data."""
    return dataset(os.path.join(directory, 'eval.tfrecords'))


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
