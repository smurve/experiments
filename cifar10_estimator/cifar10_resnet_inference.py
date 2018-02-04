"""
    Read the model from the checkpoint and do some inference
"""
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf

import cifar10_input
import cifar10_with_resnet_model
import project_utils



def main():

    #
    #  Now do some inference with the forward pass only
    #
    input_data = cifar10_input.Cifar10DataSet("/var/ellie/data/cifar10_tfr/", subset="eval", use_distortion=False)
    imgs, lbls = input_data.make_batch(10)

    model = cifar10_with_resnet_model.ResNetCifar10(
        44,
        batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5,
        is_training=False,
        data_format='channels_last')
    logits = model.forward_pass(imgs, input_data_format='channels_last')
    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits)
    }

    allvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    var_map = {"neural_network/" + v.name.split(':')[0]: v for v in allvars}
    tf.logging.set_verbosity(tf.logging.WARN)
    tf.train.init_from_checkpoint("/var/ellie/models/cifar10_new/model.ckpt-50000", var_map)
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(10):
            _lbls, _preds = sess.run([lbls, predictions['classes']])
            print(_lbls, _preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    project_utils.add_default_arguments(parser)

    args = parser.parse_args()

    if args.num_gpus < 0:
        raise ValueError(
            'Invalid GPU count: \"--num-gpus\" must be 0 or a positive integer.')
    if args.num_gpus == 0 and args.variable_strategy == 'GPU':
        raise ValueError('num-gpus=0, CPU must be used as parameter server. Set'
                         '--variable-strategy=CPU.')
    if (args.num_layers - 2) % 6 != 0:
        raise ValueError('Invalid --num-layers parameter.')
    if args.num_gpus != 0 and args.train_batch_size % args.num_gpus != 0:
        raise ValueError('--train-batch-size must be multiple of --num-gpus.')
    if args.num_gpus != 0 and args.eval_batch_size % args.num_gpus != 0:
        raise ValueError('--eval-batch-size must be multiple of --num-gpus.')

    main()
