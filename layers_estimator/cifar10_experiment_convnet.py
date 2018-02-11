"""
Experiment: MNIST without augmentation on a 2-layer convnet
Based on:
"Convolutional Neural Network Estimator for MNIST, built with tf.layers"
https://github.com/tensorflow/models/tree/master/official/mnist
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from utilities import validate_batch_size_for_multi_gpu, create_spec_provider, DefaultArgParser
import tensorflow as tf
from simple_convnet import SimpleConvnet
from cifar10_dataset import eval_input_fn, train_input_fn


#
# Create the graph and perform the training
#
def main(_):

    data_format = FLAGS.data_format
    if data_format is None:
        data_format = ('channels_first'
                       if tf.test.is_built_with_cuda() else 'channels_last')

    def get_input_tensor(features):
        """
        Just to make sure the input features are mere tensors, not dictionaries
        :param features: the input to the model function, maybe a dictionary
        :return: the extracted tensor, if it is a dictionary, else the unchanged input tensor
        """
        extr = features
        if isinstance(extr, dict):
            extr = features['image']
        return extr

    model_function = create_spec_provider(
        lambda: SimpleConvnet(data_format, width=32, height=32, channels=3, n_classes=10),
        get_input_tensor)

    if FLAGS.multi_gpu:
        print("replicating model...")
        validate_batch_size_for_multi_gpu(FLAGS.batch_size)

        model_function = tf.contrib.estimator.replicate_model_fn(
            model_function, loss_reduction=tf.losses.Reduction.MEAN)

    cifar10_classifier = tf.estimator.Estimator(
        model_fn=model_function,
        model_dir=FLAGS.model_dir,
        params={
            'data_format': data_format,
            'multi_gpu': FLAGS.multi_gpu
        })

    # Set up training hook that logs the training accuracy every 100 steps.
    tensors_to_log = {'train_accuracy': 'train_accuracy'}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)

    #
    # Perform the training and export the model
    #
    train_input = train_input_fn(FLAGS.data_dir, FLAGS.batch_size, FLAGS.train_epochs)
    cifar10_classifier.train(input_fn=train_input, hooks=[logging_hook])

    eval_input = eval_input_fn(FLAGS.data_dir, FLAGS.batch_size)
    eval_results = cifar10_classifier.evaluate(input_fn=eval_input)

    print()
    print('Evaluation results:\n\t%s' % eval_results)

    if FLAGS.export_dir is not None:
        image = tf.placeholder(tf.float32, [None, 32, 32, 3])
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'image': image,
        })
        cifar10_classifier.export_savedmodel(FLAGS.export_dir, input_fn)


if __name__ == '__main__':
    parser = DefaultArgParser()
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
