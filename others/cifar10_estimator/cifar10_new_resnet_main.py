# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ResNet model for classifying images from CIFAR-10 dataset.

Support single-host training with one or multiple devices.

ResNet as proposed in:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Deep Residual Learning for Image Recognition. arXiv:1512.03385

CIFAR-10 as in:
http://www.cs.toronto.edu/~kriz/cifar.html


"""
from __future__ import division
from __future__ import print_function

import argparse
import functools
import os

import numpy as np
import tensorflow as tf

import cifar10_input
import resnet_utils
import project_utils
import resnet_model
import towers

tf.logging.set_verbosity(tf.logging.INFO)


#
#   ResNet for CIFAR10 using my abstraction of Alex Krizhevsky's towers
#
#   Refactored form
#
def main(job_dir, data_dir, num_gpus, variable_strategy,
         use_distortion_for_training, log_device_placement, num_intra_threads,
         **hparams):
    def learning_rate(global_step):
        lr = hparams["learning_rate"]
        batch_size = hparams["train_batch_size"]
        # Suggested learning rate scheduling from
        # https://github.com/ppwwyyxx/tensorpack/blob/master/examples/ResNet/cifar10-resnet.py#L155
        num_batches_per_epoch = cifar10_input.Cifar10DataSet.num_examples_per_epoch(
            'train') // batch_size

        boundaries = [
            num_batches_per_epoch * x
            for x in np.array([82, 123, 300], dtype=np.int64)
        ]
        staged_lr = [lr * x for x in [1, 0.1, 0.01, 0.002]]

        return tf.train.piecewise_constant(global_step,
                                           boundaries, staged_lr)

    print("Starting main routine ...")
    # The env variable is on deprecation path, default is set to off.
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=log_device_placement,
        intra_op_parallelism_threads=num_intra_threads,
        gpu_options=tf.GPUOptions(force_gpu_compatible=True))

    config = resnet_utils.RunConfig(session_config=sess_config,
                                    model_dir=job_dir)

    # num_eval_examples = cifar10_input.Cifar10DataSet.num_examples_per_epoch('eval')

    train_input_fn = functools.partial(
        cifar10_input.input_fn,
        data_dir,
        subset='train',
        num_shards=num_gpus,
        batch_size=hparams["train_batch_size"],
        use_distortion_for_training=use_distortion_for_training)

    eval_input_fn = functools.partial(
        cifar10_input.input_fn,
        data_dir,
        subset='eval',
        batch_size=hparams["eval_batch_size"],
        num_shards=num_gpus)

    # experiment_function = towers.get_experiment_fn(resnet_model.resnet_model_fn,
    #                                                train_input_fn, eval_input_fn,
    #                                                num_gpus, variable_strategy,
    #                                                num_eval_examples,
    #                                                resnet_utils.device_setter_fn,
    #                                                learning_rate)

    model_fn = resnet_model.resnet_model_fn
    model_fn = towers.get_multi_tower_fn(num_gpus, variable_strategy,
                                         model_fn,
                                         resnet_utils.device_setter_fn,
                                         lr_provider=learning_rate)
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        params=hparams,
        config=config)

    train_steps = hparams['train_steps']
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=train_steps)
    exporter = tf.estimator.LatestExporter('exporter', eval_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=None, exporters=exporter)

    print("Starting... training")
    tf.estimator.train_and_evaluate(classifier, train_spec=train_spec, eval_spec=eval_spec)
    print("Done...")

    # tf.contrib.learn.learn_runner.run(
    #     experiment_function,
    #     run_config=config,
    #     hparams=tf.contrib.training.HParams(
    #         is_chief=config.is_chief,
    #         **hparams))


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

    main(**vars(args))
