import tensorflow as tf


class TensorflowDenseNet:

    def __init__(self, scope_name, seed, spec, precision=tf.float32):
        """
        WARNING: Using float16 causes this class to get stuck on CPUs
        :param spec: a list of dictionaries like {"name": ?, "size_in": ?, "size_out": ?, "activation": ?} describing
        the desired layers
        :param scope_name: name for this graph
        :param seed: A random integer seed for the weight initialization
        """

        self.precision = precision

        with tf.name_scope(scope_name):

            dim_in = spec[0]["in"]
            dim_out = spec[-1]["out"]

            with tf.name_scope('input'):
                samples = tf.placeholder(dtype=self.precision, shape=[None, dim_in], name="samples")
                labels = tf.placeholder(dtype=tf.int32, shape=[None], name="labels")
                y_ = tf.one_hot(labels, dim_out)

            # construct the sequence of dense layers
            l_in = samples
            activation = samples
            for layer in spec:
                l_dim_in = layer["in"]
                l_dim_out = layer["out"]
                l_name = layer["name"]
                l_act = layer["activation"]
                activation = self.dense(l_in, l_dim_in, l_dim_out, seed, l_name, l_act)
                l_in = activation

            a = activation

            # hidden1 = 512
            # hidden2 = 128
            # a1 = self.dense(samples, dim_in, hidden1, seed, "hidden1", tf.nn.relu)
            # a2 = self.dense(a1, hidden1, hidden2, seed, "hidden2", tf.nn.relu)
            # a = self.dense(a2, hidden2, dim_out, seed, "output", tf.nn.softmax)

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
        self.samples = samples
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
