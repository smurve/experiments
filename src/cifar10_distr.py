import argparse
import sys

import os
import tensorflow as tf
import time

from models.particular_mnist import dense_4_layers, MODEL_FILE
from models.trainer import Params
from mnist import train_batcher, test_batcher

train_batcher.set_preprocessor(lambda x: x.reshape(-1, 784))
test_batcher.set_preprocessor(lambda x: x.reshape(-1, 784))


FLAGS = None


def millies():
    return int(round(time.time() * 1000))


LOGDIR = "/var/ellie/logs"


params = Params(num_epochs=25,
                batch_size=5000,
                test_batch_size=1000,
                model_file=MODEL_FILE,
                log_dir=LOGDIR)


def main(_):

    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        with tf.device(
                tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            model = dense_4_layers()

            global_step = tf.train.get_or_create_global_step()

            bpe = 60000 / params.batch_size
            last_step = int(params.num_epochs * bpe)
            hooks = [tf.train.StopAtStepHook(last_step=last_step)]

            train_step = tf.train.AdamOptimizer(params.learning_rate) \
                .minimize(model.objective, global_step=global_step)

            init = tf.global_variables_initializer()
            i = 0

            with tf.train.MonitoredTrainingSession(master=server.target,
                                                   is_chief=(FLAGS.task_index == 0),
                                                   checkpoint_dir=LOGDIR,
                                                   save_summaries_secs=None,
                                                   save_summaries_steps=None,
                                                   hooks=hooks) as sess:

                train_writer = tf.summary.FileWriter(os.path.join(params.log_dir, "train"), sess.graph)

                now = millies()
                sess.run(init)
                while not sess.should_stop():

                    if not train_batcher.has_more():
                        train_batcher.reset()

                    if FLAGS.task_index != 0:
                        samples, labels = train_batcher.next_batch(params.batch_size)
                        _, summary = sess.run([train_step, model.summary],
                                              feed_dict={model.samples: samples, model.labels: labels})

                        train_writer.add_summary(summary, i)

                    if (i % bpe == bpe - 1) and (FLAGS.task_index == 1) and not sess.should_stop():
                        test_batcher.reset()
                        ts, tl = test_batcher.next_batch(params.test_batch_size)
                        accuracy = sess.run(model.accuracy,
                                            feed_dict={model.samples: ts, model.labels: tl})
                        duration = millies() - now
                        now = millies()
                        print("Epoch %s (%sms): Accuracy %s" %
                              (int((i+1)/bpe), duration, int(accuracy * 10000) / 100))

                    i += 1

                train_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == 'true')

    parser.add_argument("--ps_hosts", type=str, default="", help="comma-separated list of parameter server urls.")
    parser.add_argument("--worker_hosts", type=str, default="", help="comma-separated list of worker urls.")
    parser.add_argument("--job_name", type=str, default="", help="task: either 'ps' or 'worker'")
    parser.add_argument("--task_index", type=int, default=0, help="unique index for each task (p/w)")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
