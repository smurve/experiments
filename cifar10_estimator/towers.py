"""
    Tower abstraction, refactored for re-use from Alex's Estimator ResNet for CIFAR10 tutorial

    https://github.com/tensorflow/models

    See tutorials/image/cifar10_estimator when you get there

"""
from __future__ import division
from __future__ import print_function

import itertools

import six
import tensorflow as tf

import reporting_utils

tf.logging.set_verbosity(tf.logging.INFO)


def get_multi_tower_fn(num_gpus, variable_strategy,
                       model_fn, device_setter_fn, lr_provider):
    """Returns a function that will build the resnet model.
    Args:
        num_gpus: number of GPUs to use (obviously)
        variable_strategy: "GPU" or "CPU"
        model_fn: The function providing the model as in

            loss, gradvars, preds = model_fn(is_training,
                                             features,
                                             labels,
                                             data_format, params)

        lr_provider: a function that takes a tf.train.get_global_step() and returns
            a learning rate value for that step
        device_setter_fn: A device setter
    """

    def _multi_tower_model_fn(features, labels, mode, params):
        """A model function that distributes models amongst towers.

        Support single host, one or more GPU training. Parameter distribution can
        be either one of the following scheme.
        1. CPU is the parameter server and manages gradient updates.
        2. Parameters are distributed evenly across all GPUs, and the first GPU
           manages gradient updates.

        Args:
          features: a list of tensors, one for each tower
          labels: a list of tensors, one for each tower
          mode: ModeKeys.TRAIN or EVAL
          params: Hyperparameters suitable for tuning
        Returns:
          A EstimatorSpec object.
        """
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        momentum = params.momentum

        tower_features = features
        tower_labels = labels
        tower_losses = []
        tower_gradvars = []
        tower_preds = []

        # channels first (NCHW) is normally optimal on GPU and channels last (NHWC)
        # on CPU. The exception is Intel MKL on CPU which is optimal with
        # channels_last.
        data_format = params.data_format
        if not data_format:
            if num_gpus == 0:
                data_format = 'channels_last'
            else:
                data_format = 'channels_first'

        if num_gpus == 0:
            num_devices = 1
            device_type = 'cpu'
        else:
            num_devices = num_gpus
            device_type = 'gpu'

        for i in range(num_devices):
            worker_device = '/{}:{}'.format(device_type, i)

            device_setter = device_setter_fn(
                variable_strategy, worker_device, num_gpus)

            with tf.variable_scope('neural_network', reuse=bool(i != 0)):
                with tf.name_scope('tower_%d' % i) as name_scope:
                    with tf.device(device_setter):

                        loss, gradvars, preds = \
                            model_fn(is_training,
                                     tower_features[i],
                                     tower_labels[i],
                                     data_format, params)

                        # loss, gradvars, preds = \
                        #     resnet_model.resnet_model_fn(is_training,
                        #                                  tower_features[i],
                        #                                  tower_labels[i],
                        #                                  data_format, params)

                        tower_losses.append(loss)
                        tower_gradvars.append(gradvars)
                        tower_preds.append(preds)
                        if i == 0:
                            # Only trigger batch_norm moving mean and variance update from
                            # the 1st tower. Ideally, we should grab the updates from all
                            # towers but these stats accumulate extremely fast so we can
                            # ignore the other stats from the other towers without
                            # significant detriment.
                            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                           name_scope)

        # Now compute global loss and gradients.
        gradvars = []
        with tf.name_scope('gradient_averaging'):
            all_grads = {}
            for grad, var in itertools.chain(*tower_gradvars):
                if grad is not None:
                    all_grads.setdefault(var, []).append(grad)
            for var, grads in six.iteritems(all_grads):
                # Average gradients on the same device as the variables
                # to which they apply.
                with tf.device(var.device):
                    if len(grads) == 1:
                        avg_grad = grads[0]
                    else:
                        avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
                gradvars.append((avg_grad, var))

        # Device that runs the ops to apply global gradient updates.
        consolidation_device = '/gpu:0' if variable_strategy == 'GPU' else '/cpu:0'
        with tf.device(consolidation_device):

            learning_rate = lr_provider(tf.train.get_global_step())

            loss = tf.reduce_mean(tower_losses, name='loss')

            examples_sec_hook = reporting_utils.ExamplesPerSecondHook(
                params.train_batch_size, every_n_steps=10)

            tensors_to_log = {'learning_rate': learning_rate, 'loss': loss}

            logging_hook = tf.train.LoggingTensorHook(
                tensors=tensors_to_log, every_n_iter=100)

            train_hooks = [logging_hook, examples_sec_hook]

            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=momentum)

            if params.sync:
                raise ValueError("We don't support parallel processing at the moment.")
                # optimizer = tf.train.SyncReplicasOptimizer(
                #     optimizer, replicas_to_aggregate=num_workers)
                # sync_replicas_hook = optimizer.make_session_run_hook(params.is_chief)
                # train_hooks.append(sync_replicas_hook)

            train_op = [
                optimizer.apply_gradients(
                    gradvars, global_step=tf.train.get_global_step())
            ]
            # noinspection PyUnboundLocalVariable
            train_op.extend(update_ops)
            train_op = tf.group(*train_op)

            predictions = {
                'classes':
                    tf.concat([p['classes'] for p in tower_preds], axis=0),
                'probabilities':
                    tf.concat([p['probabilities'] for p in tower_preds], axis=0)
            }
            stacked_labels = tf.concat(labels, axis=0)
            metrics = {
                'accuracy':
                    tf.metrics.accuracy(stacked_labels, predictions['classes'])
            }

        # noinspection PyUnboundLocalVariable
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            training_hooks=train_hooks,
            eval_metric_ops=metrics)

    return _multi_tower_model_fn


def get_experiment_fn(model_fn,
                      train_input_fn, eval_input_fn,
                      num_gpus,
                      variable_strategy,
                      num_eval_examples,
                      device_setter_fn,
                      lr_provider):
    """Returns a multi-tower Experiment function.

    Experiments perform training on several workers in parallel,
    in other words experiments know how to invoke train and eval in a sensible
    fashion for distributed training. Arguments passed directly to this
    function are not tunable, all other arguments should be passed within
    tf.HParams, passed to the enclosed function.

    Args:
        model_fn: The function providing the model as in

            loss, gradvars, preds = model_fn(is_training,
                                             features,
                                             labels,
                                             data_format, params)

        train_input_fn: input data provder function
        eval_input_fn: eval data provider function
        num_gpus: int. Number of GPUs on each worker.
        variable_strategy: String. CPU to use CPU as the parameter server
        and GPU to use the GPUs as the parameter server.
        lr_provider: a function that provides a learning rate for a global step
        num_eval_examples: number of examples for the evaluation stop
        device_setter_fn: A device setter function
    Returns:
        A function (tf.estimator.RunConfig, tf.contrib.training.HParams) ->
        tf.contrib.learn.Experiment.

        Suitable for use by tf.contrib.learn.learn_runner, which will run various
        methods on Experiment (train, evaluate) based on information
        about the current runner in `run_config`.
    """

    def _experiment_fn(run_config, hparams):
        """Returns an Experiment."""
        if num_eval_examples % hparams.eval_batch_size != 0:
            raise ValueError(
                'validation set size must be multiple of eval_batch_size')

        train_steps = hparams.train_steps
        eval_steps = num_eval_examples // hparams.eval_batch_size

        classifier = tf.estimator.Estimator(
            model_fn=get_multi_tower_fn(num_gpus, variable_strategy,
                                        model_fn,
                                        device_setter_fn,
                                        lr_provider=lr_provider),
            config=run_config,
            params=hparams)

        return tf.contrib.learn.Experiment(
            classifier,
            train_input_fn=train_input_fn,
            eval_input_fn=eval_input_fn,
            train_steps=train_steps,
            eval_steps=eval_steps)

    return _experiment_fn
