import tensorflow as tf
import numpy as np
from helpers.gpu_utils import validate_batch_size_for_multi_gpu


def create_model_fn(model_factory, optimizer, loss_function, options, dict_key=None):
    """
    Create the model_fn to hand to the Estimator
    :param model_factory: A function that takes the model_fn params to create the model tensor
    :param optimizer: the optimizer to use for training, ignored if not running in training mode
    :param loss_function: The objective function to minimize
    :param options: an object specifying batch_size and multi_gpu usage
    :param dict_key: optional key to lookup the input tensor if 'features' is a dictionary
    :return:
    """

    def _model_fn(features, labels, mode, params):
        """The model_fn argument for creating an Estimator.
        Model functions are like factories for EstimatorSpecs

        :param features: the features, either input tensor or dictionary, if dictionary, dict_key is looked up.
        :param labels: the true labels
        :param mode: see: tf.estimator.ModeKeys
        :param params: any params
        :return: an appropriate EstimatorSpec for the chosen mode.
        """

        if options.multi_gpu:
            validate_batch_size_for_multi_gpu(options.batch_size)

        input_tensor = features
        if isinstance(input_tensor, dict):
            input_tensor = features[dict_key]

        model = model_factory(params)

        ######################################################################
        #      Spec for Prediction
        ######################################################################
        if mode == tf.estimator.ModeKeys.PREDICT:
            logits = model(input_tensor, training=False)
            predictions = {
                'classes': tf.argmax(logits, axis=1),
                'probabilities': tf.nn.softmax(logits),
            }
            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.PREDICT,
                predictions=predictions,
                export_outputs={
                    'classify': tf.estimator.export.PredictOutput(predictions)
                })

        ######################################################################
        #      Spec for Training
        ######################################################################
        if mode == tf.estimator.ModeKeys.TRAIN:

            # If we are running multi-GPU, we need to wrap the optimizer.
            # noinspection PyUnresolvedReferences
            the_optimizer = optimizer if not options.multi_gpu else \
                tf.contrib.estimator.TowerOptimizer(optimizer)

            logits = model(input_tensor, training=True)
            # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            loss = loss_function(labels=labels, logits=logits)
            accuracy = tf.metrics.accuracy(
                labels=labels, predictions=tf.argmax(logits, axis=1))
            # Name the accuracy tensor 'train_accuracy' to demonstrate the
            # LoggingTensorHook.
            tf.identity(accuracy[1], name='train_accuracy')
            tf.summary.scalar('train_accuracy', accuracy[1])

            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.TRAIN,
                loss=loss,
                train_op=the_optimizer.minimize(loss, tf.train.get_or_create_global_step()))

        ######################################################################
        #      Spec for Evaluation
        ######################################################################
        if mode == tf.estimator.ModeKeys.EVAL:
            logits = model(input_tensor, training=False)
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
        
    ######################################################################
    #      If multi-gpu, wrap into map-reduce replicator
    ######################################################################
    # noinspection PyUnresolvedReferences
    return _model_fn if not options.multi_gpu else tf.contrib.estimator.replicate_model_fn(
        _model_fn, loss_reduction=tf.losses.Reduction.MEAN)


def split_datasource(ds, num_records, ratio):
    """
    :param ds: the dataset to be split
    :param num_records: the number of records in the dataset
    :param ratio: the ratio of records in the first part
    :return a pair of datasources together equivalent to the original datasource
    """
    # noinspection PyUnresolvedReferences
    idx = np.array(range(num_records))
    idx_ds = tf.data.Dataset.from_tensor_slices(tf.constant(idx))
    ds_i = tf.data.Dataset.zip((ds, idx_ds))
    ds1 = ds_i.filter(lambda x, y: y < int(num_records * ratio)).map(lambda x, y: x)
    ds2 = ds_i.filter(lambda x, y: y >= int(num_records * ratio)).map(lambda x, y: x)
    return ds1, ds2
