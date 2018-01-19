from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from models.model import TensorflowSoftmaxRegression
# from models.model import TensorflowHiddenx2Feedforward
from batchers import MnistBatcher
import tensorflow as tf
import os


SEED = 123
SCOPE = "mnist_softmax_regression"
DATA_DIR = "/var/ellie/data/mnist"
MODEL_DIR = "/var/ellie/models/mnist"
MODEL_FILE = os.path.join(MODEL_DIR, "model.cpkt")
img_train = os.path.join(DATA_DIR, "train-images-idx3-ubyte.gz")
lbl_train = os.path.join(DATA_DIR, "train-labels-idx1-ubyte.gz")
img_test = os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz")
lbl_test = os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz")
train_batcher = MnistBatcher(img_file=img_train, lbl_file=lbl_train, num_samples=60000)
test_batcher = MnistBatcher(img_file=img_test, lbl_file=lbl_test, num_samples=10000)
NUM_EPOCHS = 200
BATCH_SIZE = 256
TEST_BATCH_SIZE = 1000
model = TensorflowSoftmaxRegression(SCOPE, SEED, 784, 10, precision=tf.float32)
# model = TensorflowHiddenx2Feedforward(SCOPE, SEED)

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    i = 0
    train_writer = tf.summary.FileWriter("./train", sess.graph)
    test_writer = tf.summary.FileWriter("./test")
    train_step = tf.train.AdamOptimizer().minimize(model.objective)
    sess.run(tf.global_variables_initializer())
    for epoch in range(NUM_EPOCHS):
        test_batcher.reset()

        images, labels = test_batcher.next_batch(TEST_BATCH_SIZE)
        accuracy, summary = sess.run([model.accuracy, model.summary],
                                     feed_dict={model.images: images.reshape(-1, 784), model.labels: labels})
        test_writer.add_summary(summary, i)
        print("Test Accuracy after batch %s: %s" % (i, accuracy))

        train_batcher.reset()
        while train_batcher.has_more():
            images, labels = train_batcher.next_batch(BATCH_SIZE)
            images = images.reshape(-1, 784)
            _, summary = sess.run([train_step, model.summary],
                                  feed_dict={model.images: images, model.labels: labels})
            train_writer.add_summary(summary, i)
            i += 1

    saver = tf.train.Saver()
    saver.save(sess, MODEL_FILE)

    train_writer.close()
    test_writer.close()
