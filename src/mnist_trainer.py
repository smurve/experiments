from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.particular_mnist import dense_4_layers
from models.particular_mnist import MODEL_FILE
from models.trainer import Params, train
from mnist import train_batcher, test_batcher
import tensorflow as tf
import os

train_batcher.set_preprocessor(lambda x: x.reshape(-1, 784))
test_batcher.set_preprocessor(lambda x: x.reshape(-1, 784))


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
    device_count={'GPU': 2},
    intra_op_parallelism_threads=12,
    inter_op_parallelism_threads=12)

LOGDIR = "/var/ellie/models/mnist"

params = Params(num_epochs=100,
                batch_size=10000,
                test_batch_size=1000,
                model_file=MODEL_FILE,
                log_dir=LOGDIR)

clean_tensorboard_logs(LOGDIR)

model = dense_4_layers()
train(model, train_batcher, test_batcher, config, params)
