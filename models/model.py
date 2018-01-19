import tensorflow as tf


class TensorflowSoftmaxRegression:

    def __init__(self, scope_name, seed, dim_in, dim_out, precision=tf.float32):
        """
        WARNING: Using float16 causes this class to get stuck on CPUs

        :param scope_name: name for this graph
        :param seed: A random integer seed for the weight initialization
        """

        self.precision = precision

        with tf.name_scope(scope_name):

            with tf.name_scope('input'):
                images = tf.placeholder(dtype=self.precision, shape=[None, dim_in], name="images")
                labels = tf.placeholder(dtype=tf.int32, shape=[None], name="labels")
                y_ = tf.one_hot(labels, dim_out)

            a = self.dense(images, dim_in, dim_out, seed, "layer1", tf.nn.softmax)

            with tf.name_scope("objective"):
                objective = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=a),
                                           name="objective")
            tf.summary.scalar('cross_entropy', objective)

            with tf.name_scope("correct_predictions"):
                correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(y_, 1))

            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, self.precision))
            tf.summary.scalar('accuracy', accuracy)

        merged = tf.summary.merge_all()

        self.summary = merged
        self.images = images
        self.labels = labels
        self.activation = a
        self.objective = objective
        self.accuracy = accuracy

    @staticmethod
    def variable_summary(var, scope):
        """
        @type var: Variable
        @param var: Variable to collect summaries of
        @param scope: A visible name of the variable
        """
        with tf.name_scope(scope):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def weight_variable(self, shape, seed, mean=0.0, stddev=0.1):
        rnd_w = tf.truncated_normal(shape, mean=mean, stddev=stddev, dtype=self.precision, seed=seed)
        w = tf.Variable(rnd_w, name="w", dtype=self.precision)
        self.variable_summary(w, "summary_W")
        return w

    def bias_variable(self, shape):
        b = tf.Variable(tf.zeros(shape, dtype=self.precision), dtype=self.precision, name="b")
        self.variable_summary(b, "summary_b")
        return b

    def dense(self, inp, dim_in, dim_out, seed, scope, activation):
        with tf.name_scope(scope):
            w = self.weight_variable([dim_in, dim_out], seed)
            b = self.bias_variable([dim_out])
            z = tf.matmul(inp, w) + b
            a = activation(z)
        return a


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
