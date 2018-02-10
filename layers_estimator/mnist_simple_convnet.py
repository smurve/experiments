"""
    A 2-layer convolutional network model 
    Based on:
    "Convolutional Neural Network Estimator for MNIST, built with tf.layers"
    https://github.com/tensorflow/models/tree/master/official/mnist
"""
import tensorflow as tf


class Model(object):
    """Class that defines a graph to recognize digits in the MNIST dataset."""

    def __init__(self, data_format):
        """Creates a model for classifying a hand-written digit.

        Args:
          data_format: Either 'channels_first' or 'channels_last'.
            'channels_first' is typically faster on GPUs while 'channels_last' is
            typically faster on CPUs. See
            https://www.tensorflow.org/performance/performance_guide#data_formats
        """
        width = 28
        height = 28
        if data_format == 'channels_first':
            self._input_shape = [-1, 1, width, height]
        else:
            assert data_format == 'channels_last'
            self._input_shape = [-1, width, height, 1]

        self.conv1 = tf.layers.Conv2D(
            32, 5, padding='same', data_format=data_format, activation=tf.nn.relu)
        self.conv2 = tf.layers.Conv2D(
            64, 5, padding='same', data_format=data_format, activation=tf.nn.relu)
        self.fc1 = tf.layers.Dense(1024, activation=tf.nn.relu)
        self.fc2 = tf.layers.Dense(10)
        self.dropout = tf.layers.Dropout(0.4)
        self.max_pool2d = tf.layers.MaxPooling2D(
            (2, 2), (2, 2), padding='same', data_format=data_format)

    def fwd_pass(self, inputs, training):
        """The model's forward pass

        Args:
          inputs: A Tensor representing a batch of input images.
          training: A boolean. Set to True to add operations required only when
            training the classifier.

        Returns:
          A pair of two tensors
          logits and probabilities with shape [<batch_size>, 10].
        """
        y = tf.reshape(inputs, self._input_shape)
        y = self.conv1(y)
        y = self.max_pool2d(y)
        y = self.conv2(y)
        y = self.max_pool2d(y)
        y = tf.layers.flatten(y)
        y = self.fc1(y)
        y = self.dropout(y, training=training)
        logits = self.fc2(y)

        return logits, tf.nn.softmax(logits)
