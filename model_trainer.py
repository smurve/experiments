# from models.model import TensorflowMnistSoftmaxRegression
from models.model import TensorflowHiddenx2Feedforward
from batchers import MnistBatcher
import tensorflow as tf
import os


SEED = 123
MODEL_FILE = os.path.join(os.environ.get("TMPDIR", "./"), "model.cpkt")
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
NUM_EPOCHS = 300
BATCH_SIZE = 256
# model = TensorflowMnistSoftmaxRegression(SCOPE, SEED)
model = TensorflowHiddenx2Feedforward(SCOPE, SEED)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    train_step = tf.train.AdamOptimizer().minimize(model.objective)
    sess.run(tf.global_variables_initializer())
    for epoch in range(NUM_EPOCHS):
        train_batcher.reset()
        if (epoch % 10) == 0:
            print("Epoch Nr. %s" % (epoch + 1))
            test_batcher.reset()
            test_images, test_labels = test_batcher.next_batch(200)
            test_images = test_images.reshape(-1, 784)
            test_accuracy = sess.run(model.accuracy,
                                     feed_dict={model.images: test_images, model.labels: test_labels})
            test_accuracy = round(test_accuracy * 100, 2)
            print("Test Accuracy: %s" % test_accuracy)

            train_images, train_labels = train_batcher.next_batch(200)
            train_images = train_images.reshape(-1, 784)
            train_accuracy = sess.run(model.accuracy,
                                      feed_dict={model.images: train_images, model.labels: train_labels})
            train_accuracy = round(train_accuracy * 100, 2)
            print("Train Accuracy: %s" % train_accuracy)
            train_batcher.reset()

        while train_batcher.has_more():
            images, labels = train_batcher.next_batch(BATCH_SIZE)
            images = images.reshape(-1, 784)
            train_step.run(session=sess, feed_dict={model.images: images, model.labels: labels})

    saver = tf.train.Saver()
    saver.save(sess, MODEL_FILE)
