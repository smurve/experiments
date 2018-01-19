from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from models.model import TensorflowDenseNet
from batchers import MnistBatcher
import tensorflow as tf
import os
import time


DATA_DIR = "/var/ellie/data/mnist"
MODEL_DIR = "/var/ellie/models/mnist"
MODEL_FILE = os.path.join(MODEL_DIR, "model.ckpt")
img_train = os.path.join(DATA_DIR, "train-images-idx3-ubyte.gz")
lbl_train = os.path.join(DATA_DIR, "train-labels-idx1-ubyte.gz")
img_test = os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz")
lbl_test = os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz")
train_batcher = MnistBatcher(img_file=img_train, lbl_file=lbl_train, num_samples=60000)
test_batcher = MnistBatcher(img_file=img_test, lbl_file=lbl_test, num_samples=10000)
NUM_EPOCHS = 50
BATCH_SIZE = 2000
TEST_BATCH_SIZE = 1000
SEED = 123
SCOPE = "mnist_softmax_regression"
spec = [
    {"name": "input", "in": 784, "out": 1024, "activation": tf.nn.relu},
    {"name": "hidden1", "in": 1024, "out": 128, "activation": tf.nn.relu},
    {"name": "hidden2", "in": 128, "out": 10, "activation": tf.nn.softmax}]

model = TensorflowDenseNet(SCOPE, SEED, spec)


def millies():
    return int(round(time.time() * 1000))


config = tf.ConfigProto(
    log_device_placement=False,
    device_count={'CPU': 12},
    intra_op_parallelism_threads=12,
    inter_op_parallelism_threads=12)

with tf.Session(config=config) as sess:
    i = 0
    train_writer = tf.summary.FileWriter(os.path.join(MODEL_DIR, "train"), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(MODEL_DIR, "./test"))
    train_step = tf.train.AdamOptimizer().minimize(model.objective)
    sess.run(tf.global_variables_initializer())
    now = millies()
    for epoch in range(NUM_EPOCHS):
        test_batcher.reset()

        samples, labels = test_batcher.next_batch(TEST_BATCH_SIZE)
        samples = samples.reshape(-1, 784)
        accuracy, summary = sess.run([model.accuracy, model.summary],
                                     feed_dict={model.samples: samples, model.labels: labels})
        test_writer.add_summary(summary, i)
        duration = millies()-now
        if epoch > 0:
            print("Epoch %s took %sms. Accuracy now: %s" % (epoch, duration, accuracy))
        now = millies()

        train_batcher.reset()
        while train_batcher.has_more():
            samples, labels = train_batcher.next_batch(BATCH_SIZE)
            samples = samples.reshape(-1, 784)
            _, summary = sess.run([train_step, model.summary],
                                  feed_dict={model.samples: samples, model.labels: labels})
            train_writer.add_summary(summary, i)
            i += 1

    saver = tf.train.Saver()
    saver.save(sess, MODEL_FILE)

    train_writer.close()
    test_writer.close()
