"""
    Utility functions based on:
    "Convolutional Neural Network Estimator for MNIST, built with tf.layers"
    https://github.com/tensorflow/models/tree/master/official/mnist
"""
import tensorflow as tf
import argparse


def validate_batch_size_for_multi_gpu(batch_size):
    """For multi-gpu, batch-size must be a multiple of the number of
    available GPUs.

    Note that this should eventually be handled by replicate_model_fn
    directly. Multi-GPU support is currently experimental, however,
    so doing the work here until that feature is in place.
    """
    from tensorflow.python.client import device_lib

    local_device_protos = device_lib.list_local_devices()
    num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
    if not num_gpus:
        raise ValueError('Multi-GPU mode was specified, but no GPUs '
                         'were found. To use CPU, run without --multi_gpu.')

    remainder = batch_size % num_gpus
    if remainder:
        err = ('When running with multiple GPUs, batch size '
               'must be a multiple of the number of available GPUs. '
               'Found {} GPUs with a batch size of {}; try --batch_size={} instead.'
               ).format(num_gpus, batch_size, batch_size - remainder)
        raise ValueError(err)


def create_spec_provider(model_factory, pre_process=lambda x: x):
    """
    A special function that can creates a spec provider function appropriate
    for use in the Estimator's constructor. Call it like:

        model_fn = create_spec_provider(lambda: Model(...), processor)
        est = Estimator ( model_fn, ... )

    Note the lambda, that's necessary, because the model can only be created in the
    estimator's graph context. So we tell the estimator how, and let it do the job when the time is come.

    :param model_factory: a lamba that takes no argument and returns a Model
    :param pre_process: A function to pre-process the raw features, default to the identity
    :return: a spec provider function that returns appropriate EstimatorSpecs
    """
    def spec_provider(features, labels, mode, params):
        """
            The model_fn argument for creating an Estimator.
        """

        # The model's graph must be created within this function
        model = model_factory()

        image = pre_process(features)
        # image = features
        # if isinstance(image, dict):
        #     image = features['image']

        if mode == tf.estimator.ModeKeys.PREDICT:
            logits, probs = model.fwd_pass(image, training=False)
            predictions = {
                'classes': tf.argmax(logits, axis=1),
                'probabilities': probs,
            }
            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.PREDICT,
                predictions=predictions,
                export_outputs={
                    'classify': tf.estimator.export.PredictOutput(predictions)
                })

        elif mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

            # If we are running multi-GPU, we need to wrap the optimizer.
            if params.get('multi_gpu'):
                print("Using Tower Optimizer...")
                optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

            logits, _ = model.fwd_pass(image, training=True)

            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

            accuracy = tf.metrics.accuracy(
                labels=labels, predictions=tf.argmax(logits, axis=1))

            # Name the accuracy tensor 'train_accuracy' to demonstrate the
            # LoggingTensorHook.
            tf.identity(accuracy[1], name='train_accuracy')
            tf.summary.scalar('train_accuracy', accuracy[1])

            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.TRAIN,
                loss=loss,
                train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))

        elif mode == tf.estimator.ModeKeys.EVAL:
            logits, _ = model.fwd_pass(image, training=False)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.EVAL,
                loss=loss,
                eval_metric_ops={
                    'accuracy':
                        tf.metrics.accuracy(
                            labels=labels,
                            predictions=tf.argmax(logits, axis=1)),
                })

    return spec_provider


class DefaultArgParser(argparse.ArgumentParser):

    def __init__(self):
        super(DefaultArgParser, self).__init__()

        self.add_argument(
            '--multi_gpu', action='store_true',
            help='If set, run across all available GPUs.')
        self.add_argument(
            '--batch_size',
            type=int,
            default=128,
            help='Number of images to process in a batch')
        self.add_argument(
            '--data_dir',
            type=str, required=True,
            help='Path to directory containing the input data')
        self.add_argument(
            '--model_dir',
            type=str, required=True,
            help='The directory where the model will be stored.')
        self.add_argument(
            '--train_epochs',
            type=int, required=True,
            help='Number of epochs to train.')
        self.add_argument(
            '--data_format',
            type=str,
            default=None,
            choices=['channels_first', 'channels_last'],
            help='A flag to override the data format used in the model. '
                 'channels_first provides a performance boost on GPU but is not always '
                 'compatible with CPU. If left unspecified, the data format will be '
                 'chosen automatically based on whether TensorFlow was built for CPU or '
                 'GPU.')
        self.add_argument(
            '--export_dir',
            type=str, required=True,
            help='The directory where the exported SavedModel will be stored.')
