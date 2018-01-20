from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.particular import dense_4_layers
from models.particular import MODEL_FILE
from models.trainer import Params, train
from mnist import train_batcher, test_batcher
import tensorflow as tf

train_batcher.set_preprocessor(lambda x: x.reshape(-1, 784))
test_batcher.set_preprocessor(lambda x: x.reshape(-1, 784))

config = tf.ConfigProto(
    log_device_placement=False,
    device_count={'CPU': 12},
    intra_op_parallelism_threads=12,
    inter_op_parallelism_threads=12)

params = Params(num_epochs=50,
                batch_size=2000,
                test_batch_size=1000,
                model_file=MODEL_FILE,
                log_dir="/var/ellie/models/mnist")

model = dense_4_layers()
train(model, train_batcher, test_batcher, config, params)
