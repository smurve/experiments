import tensorflow as tf


class TensorflowMnistSoftmaxRegression:
    """
    A Model encapsulating
    """

    def __init__(self, scope_name, seed):
        """
        :param scope_name: name for this graph
        :param seed: A random integer seed for the weight initialization
        """
        with tf.name_scope(scope_name):
            images = tf.placeholder(tf.float32, shape=[None, 784], name="images")
            labels = tf.placeholder(tf.int32, shape=[None], name="labels")
            y_ = tf.one_hot(labels, 10)
            rnd_w = tf.truncated_normal([784, 10], mean=0.0, stddev=0.1, dtype=tf.float32, seed=seed)
            w = tf.Variable(rnd_w, name="w")
            b = tf.Variable(tf.zeros([10]), dtype=tf.float32, name="b")
            z = tf.matmul(images, w) + b
            a = tf.nn.softmax(z, name="activation")
            objective = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=a),
                                       name="objective")
        self.images = images
        self.labels = labels
        self.activation = a
        self.objective = objective
