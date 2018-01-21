from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.particular_cifar10 import dense_3_layers
from models.particular_cifar10 import MODEL_FILE
from models.trainer import Params, train
from cifar10.batcher import CIFAR10Batcher
import tensorflow as tf
import os

train_batcher = CIFAR10Batcher()
test_batcher = CIFAR10Batcher(['test_batch'])


def clean_tensorboard_logs(logdir):
    for subdir in ['test', 'train']:
        path = os.path.join(logdir, subdir)
        if os.path.isdir(path):
            files = os.listdir(path)
            for file in files:
                fpath = os.path.join(path, file)
                os.unlink(fpath)


config = tf.ConfigProto(
    log_device_placement=False,
    device_count={'CPU': 12},
    intra_op_parallelism_threads=12,
    inter_op_parallelism_threads=12)

LOGDIR = "/var/ellie/models/cifar10"

params = Params(num_epochs=200,
                batch_size=2000,
                test_batch_size=1000,
                model_file=MODEL_FILE,
                log_dir=LOGDIR,
                learning_rate=3e-5)

clean_tensorboard_logs(LOGDIR)

model = dense_3_layers()
train(model, train_batcher, test_batcher, config, params)
