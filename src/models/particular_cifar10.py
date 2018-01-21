import tensorflow as tf
import os
from .model import TensorflowDenseNet

MODEL_DIR = "/var/ellie/models/cifar10"
MODEL_FILE = os.path.join(MODEL_DIR, "model.ckpt")

SEED = 123
SCOPE = "cifar10_softmax_regression"
spec = [
    {"name": "input", "in": 3072, "out": 256, "activation": tf.nn.relu},
    {"name": "hidden1", "in": 256, "out": 10, "activation": tf.nn.softmax}]


def dense_3_layers():
    return TensorflowDenseNet(SCOPE, SEED, spec)
