import tensorflow as tf
import os
import time


def millies():
    return int(round(time.time() * 1000))


class Params:
    def __init__(self, num_epochs, batch_size, test_batch_size, model_file, log_dir, learning_rate=1e-3):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.model_file = model_file
        self.log_dir = log_dir
        self.learning_rate = learning_rate


def train(model, train_batcher, test_batcher, config, params):
    with tf.Session(config=config) as sess:
        i = 0
        train_writer = tf.summary.FileWriter(os.path.join(params.log_dir, "train"), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(params.log_dir, "test"))
        train_step = tf.train.AdamOptimizer(params.learning_rate).minimize(model.objective)
        sess.run(tf.global_variables_initializer())
        now = millies()
        for epoch in range(params.num_epochs):
            test_batcher.reset()

            samples, labels = test_batcher.next_batch(params.test_batch_size)
            accuracy, summary = sess.run([model.accuracy, model.summary],
                                         feed_dict={model.samples: samples, model.labels: labels})
            test_writer.add_summary(summary, i)
            duration = millies()-now
            if epoch > 0:
                print("Epoch %s took %sms. Accuracy now: %s" % (epoch, duration, accuracy))
            now = millies()

            train_batcher.reset()
            while train_batcher.has_more():
                samples, labels = train_batcher.next_batch(params.batch_size)
                _, summary = sess.run([train_step, model.summary],
                                      feed_dict={model.samples: samples, model.labels: labels})
                train_writer.add_summary(summary, i)
                i += 1

        saver = tf.train.Saver()
        saver.save(sess, params.model_file)

        train_writer.close()
        test_writer.close()


def train2(model, training_data, config, params):
    with tf.Session(config=config) as sess:
        i = 0
        train_writer = tf.summary.FileWriter(os.path.join(params.log_dir, "train"), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(params.log_dir, "test"))
        train_step = tf.train.AdamOptimizer(params.learning_rate).minimize(model.objective)
        sess.run(tf.global_variables_initializer())

        samples, labels = training_data
        accuracy, summary = sess.run([model.accuracy, model.summary],
                                     feed_dict={model.samples: samples, model.labels: labels})
        test_writer.add_summary(summary, i)

        _, summary = sess.run([train_step, model.summary],
                              feed_dict={model.samples: samples, model.labels: labels})
        train_writer.add_summary(summary, i)
        i += 1

        saver = tf.train.Saver()
        saver.save(sess, params.model_file)

        train_writer.close()
        test_writer.close()
