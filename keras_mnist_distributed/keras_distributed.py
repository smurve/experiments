"""
The code is inspired from François Chollet's answer to the following quora question[1]
and distributed tensorflow tutorial[2].

It runs the Keras MNIST mlp example across multiple servers.

This sample code runs multiple processes on a single host. It can be configured
to run on multiple hosts simply by chaning the host names given in *ClusterSpec*.

Training the model:

Start the parameter server
  python keras_distributed.py --job_name="ps" --task_index=0

Start the three workers
  python keras_distributed.py --job_name="worker" --task_index=0
  python keras_distributed.py --job_name="worker" --task_index=1
  python keras_distributed.py --job_name="worker" --task_index=2

[1] https://www.quora.com/What-is-the-state-of-distributed-learning-multi-GPU-and-across-\
    multiple-hosts-in-Keras-and-what-are-the-future-plans
[2] https://www.tensorflow.org/deploy/distributed
"""

import tensorflow as tf
import keras

# Define input flags to identify the job and task
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

# Create a tensorflow cluster
# Replace localhost with the host names if you are running on multiple hosts
cluster = tf.train.ClusterSpec({"ps": ["scylla:2222"],
                                "worker": ["scylla:2223",
                                           "charybdis:2224",
                                           "charybdis:2225"]})

# Start the server
server = tf.train.Server(cluster,
                         job_name=FLAGS.job_name,
                         task_index=FLAGS.task_index)

# Configurations
batch_size = 128
learning_rate = 0.0005
training_iterations = 100
num_classes = 10
log_frequency = 10


# Load mnist data
def load_data():
    global mnist
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    print("Data loaded")


# Create Keras model
def create_model():
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    seq = Sequential()
    seq.add(Dense(512, activation='relu', input_shape=(784,)))
    seq.add(Dropout(0.2))
    seq.add(Dense(512, activation='relu'))
    seq.add(Dropout(0.2))
    seq.add(Dense(10, activation='softmax'))

    seq.summary()

    return seq


# Create the optimizer
# We cannot use model.compile and model.fit
def create_optimizer(model, targets):
    preds = model.output
    loss = tf.reduce_mean(
        keras.losses.categorical_crossentropy(targets, preds))

    # Only if you have regularizers, not in this example
    _total_loss = loss * 1.0  # Copy
    for regularizer_loss in model.losses:
        tf.assign_add(_total_loss, regularizer_loss)

    optimizer = tf.train.RMSPropOptimizer(learning_rate)

    # Barrier to compute gradients after updating moving avg of batch norm
    with tf.control_dependencies(model.updates):
        barrier = tf.no_op(name="update_barrier")

    with tf.control_dependencies([barrier]):
        grads = optimizer.compute_gradients(
            _total_loss,
            model.trainable_weights)
        grad_updates = optimizer.apply_gradients(grads)

    with tf.control_dependencies([grad_updates]):
        _train_op = tf.identity(_total_loss, name="train")

    return _train_op, _total_loss, preds


# Train the model (a single step)
def train(_train_op, _total_loss, _global_step, _step):
    import time
    start_time = time.time()
    batch_x, batch_y = mnist.train.next_batch(batch_size)

    # perform the operations we defined earlier on batch
    loss_value, step_value = sess.run(
        [_train_op, _global_step],
        feed_dict={
            mnist_mlp.inputs[0]: batch_x,
            labels: batch_y})

    if _step % log_frequency == 0:
        elapsed_time = time.time() - start_time
        accuracy = sess.run(_total_loss,
                            feed_dict={
                                mnist_mlp.inputs[0]: mnist.test.images,
                                labels: mnist.test.labels})
        print("Step: %d," % (step_value + 1),
              " Iteration: %2d," % _step,
              " Cost: %.4f," % loss_value,
              " Accuracy: %.4f" % accuracy,
              " AvgTime: %3.2fms" % float(elapsed_time * 1000 / log_frequency))


if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    load_data()

    # Assign operations to local server
    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):
        keras.backend.set_learning_phase(1)
        keras.backend.manual_variable_initialization(True)
        mnist_mlp = create_model()
        labels = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")
        train_op, total_loss, predictions = create_optimizer(mnist_mlp, labels)

        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        init_op = tf.global_variables_initializer()

    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             global_step=global_step,
                             logdir="/tmp/train_logs",
                             save_model_secs=600,
                             init_op=init_op)

    print("Waiting for other servers")
    with sv.managed_session(server.target) as sess:
        keras.backend.set_session(sess)
        step = 0
        while not sv.should_stop() and step < 1000000:
            train(train_op, total_loss, global_step, step)
            step += 1

    sv.stop()
    print("done")
