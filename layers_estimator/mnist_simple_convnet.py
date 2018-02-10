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
        self.data_format = data_format
        width = 28
        height = 28
        if data_format == 'channels_first':
            self._input_shape = [-1, 1, width, height]
        else:
            assert data_format == 'channels_last'
            self._input_shape = [-1, width, height, 1]

    @staticmethod
    def dense(name, outputsize):
        return tf.layers.Dense(outputsize, activation=tf.nn.relu, name=name)

    def maxpool(self, name):
        return tf.layers.MaxPooling2D((2, 2), (2, 2), padding='same',
                                      data_format=self.data_format, name=name)

    def conv(self, name, n_fields, size_fields):
        return tf.layers.Conv2D(n_fields, size_fields, padding='same',
                                data_format=self.data_format, activation=tf.nn.relu, name=name)

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

        y = self.conv(n_fields=32, size_fields=5, name="conv1")(y)
        y = self.maxpool(name="max1")(y)

        y = self.conv(n_fields=64, size_fields=5, name="conv1")(y)
        y = self.maxpool(name="max2")(y)

        y = self.conv(n_fields=128, size_fields=5, name="conv1")(y)
        y = self.maxpool(name="max3")(y)

        y = tf.layers.flatten(y)

        y = self.dense(outputsize=2048, name="dense1")(y)
        y = tf.layers.Dropout(0.4)(y, training=training)

        y = self.dense(outputsize=256, name="dense1")(y)
        y = tf.layers.Dropout(0.4)(y, training=training)

        logits = tf.layers.Dense(10)(y)

        return logits, tf.nn.softmax(logits)
