import tensorflow as tf


class Model:
    """Model to recognize digits in the MNIST dataset.

        Naive 3-layer feed forward network
    """

    # noinspection PyUnusedLocal
    def __init__(self, params, inp=784, hidden1=1024, hidden2=128, output=10):
        """
        Constructor: produces three dense layer functions with dimensions
        input->hidden1, hidden1->hidden2, hidden2->output
        Args:
            params: not used in this simple model.
            inp: dimension of the input layer
            hidden1: dimension of the 1st hidden layer
            hidden2: dimension of the 2nd hidden layer
            output: dimension of the output layer
        """

        self._input_shape = [-1, inp]

        self.fc1 = tf.layers.Dense(hidden1, activation=tf.nn.relu)
        self.fc2 = tf.layers.Dense(hidden2, activation=tf.nn.relu)
        self.fc3 = tf.layers.Dense(output)

    def __call__(self, inputs, training):
        """Add operations to classify a batch of input images.

        Args:
          inputs: A Tensor representing a batch of input images.
          training: A boolean. Set to True to add operations required only when
            training the classifier.

        Returns:
          A logits Tensor with shape [<batch_size>, 10].
        """
        y = tf.reshape(inputs, self._input_shape)
        y = self.fc1(y)
        y = self.fc2(y)
        return self.fc3(y)
