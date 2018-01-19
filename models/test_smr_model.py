from .model import TensorflowSoftmaxRegression
from batchers import MnistBatcher
import tensorflow as tf
import os
import numpy as np


class TestModel:

    SEED = 123
    MODEL_FILE = os.path.join(os.environ.get("TMPDIR", "./"), "model.cpkt")
    SCOPE = "mnist_softmax_regression"
    img_fiile = './testdata/t10k-images-idx3-ubyte.gz'
    lbl_file = './testdata/t10k-labels-idx1-ubyte.gz'
    batcher = MnistBatcher(img_file=img_fiile, lbl_file=lbl_file, num_samples=30)

    def test_restore(self):
        self.batcher.reset()
        batch, _ = self.batcher.next_batch(4)
        batch = batch.reshape(-1, 784)
        tf.reset_default_graph()
        model = TensorflowSoftmaxRegression(self.SCOPE, self.SEED, 784, 10)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            res1 = sess.run(model.activation, feed_dict={model.images: batch})
            saver = tf.train.Saver()
            saver.save(sess, self.MODEL_FILE)

        tf.reset_default_graph()
        model = TensorflowSoftmaxRegression(self.SCOPE, self.SEED, 784, 10)
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.MODEL_FILE)
            res2 = sess.run(model.activation, feed_dict={model.images: batch})

        assert np.array_equal(res1, res2)

    def test_train(self):
        """
        train with the first 1000 images
        """
        with tf.Session() as sess:
            model = TensorflowSoftmaxRegression(self.SCOPE, self.SEED, 784, 10)
            sess.run(tf.global_variables_initializer())
            train_step = tf.train.GradientDescentOptimizer(.3).minimize(model.objective)
            self.batcher.reset()
            images, labels = self.batcher.next_batch(1000)
            x = images.reshape(-1, 784)
            for _ in range(1000):
                train_step.run(session=sess, feed_dict={model.images: x, model.labels: labels})

            # see the classification succeed for the first 5 images
            self.batcher.reset()
            img, _ = self.batcher.next_batch(5)
            img = img.reshape(-1, 784)
            res = sess.run(model.activation, feed_dict={model.images: img})
            assert np.array_equal(np.argmax(res, 1), [7, 2, 1, 0, 4])
