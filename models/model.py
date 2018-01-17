import tensorflow as tf


class TensorflowMnistSoftmaxRegression:

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

            correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.images = images
        self.labels = labels
        self.activation = a
        self.objective = objective
        self.accuracy = accuracy


class TensorflowHiddenx2Feedforward:

    def __init__(self, scope_name, seed):
        """
        :param scope_name: name for this graph
        :param seed: A random integer seed for the weight initialization
        """
        with tf.name_scope(scope_name):
            images = tf.placeholder(tf.float32, shape=[None, 784], name="images")
            labels = tf.placeholder(tf.int32, shape=[None], name="labels")
            y_ = tf.one_hot(labels, 10)
            rnd_w1 = tf.truncated_normal([784, 256], mean=0.0, stddev=0.1, dtype=tf.float32, seed=seed)
            w1 = tf.Variable(rnd_w1, name="w1")
            b1 = tf.Variable(tf.zeros([256]), dtype=tf.float32, name="b1")
            z1 = tf.matmul(images, w1) + b1
            a1 = tf.nn.relu(z1, name="activation1")

            rnd_w2 = tf.truncated_normal([256, 64], mean=0.0, stddev=0.1, dtype=tf.float32, seed=seed)
            w2 = tf.Variable(rnd_w2, name="w2")
            b2 = tf.Variable(tf.zeros([64]), dtype=tf.float32, name="b2")
            z2 = tf.matmul(a1, w2) + b2
            a2 = tf.nn.softmax(z2, name="activation2")

            rnd_w3 = tf.truncated_normal([64, 10], mean=0.0, stddev=0.1, dtype=tf.float32, seed=seed)
            w3 = tf.Variable(rnd_w3, name="w3")
            b3 = tf.Variable(tf.zeros([10]), dtype=tf.float32, name="b3")
            z3 = tf.matmul(a2, w3) + b3
            a3 = tf.nn.relu(z3, name="activation3")

            objective = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=a3),
                                       name="objective")

            correct_prediction = tf.equal(tf.argmax(a3, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.images = images
        self.labels = labels
        self.activation = a3
        self.objective = objective
        self.accuracy = accuracy
