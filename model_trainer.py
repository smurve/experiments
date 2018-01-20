from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.particular import dense_4_layers as model
from models.particular import MODEL_FILE
from models.trainer import Params, train
from mnist import train_batcher, test_batcher
import tensorflow as tf

train_batcher.set_preprocessor(lambda x: x.reshape(-1, 784))
test_batcher.set_preprocessor(lambda x: x.reshape(-1, 784))

LOG_DIR = "/var/ellie/models/mnist"
NUM_EPOCHS = 5
BATCH_SIZE = 2000
TEST_BATCH_SIZE = 1000

config = tf.ConfigProto(
    log_device_placement=False,
    device_count={'CPU': 12},
    intra_op_parallelism_threads=12,
    inter_op_parallelism_threads=12)

params = Params(NUM_EPOCHS, BATCH_SIZE, TEST_BATCH_SIZE, MODEL_FILE, LOG_DIR)

train(model, train_batcher, test_batcher, config, params)
