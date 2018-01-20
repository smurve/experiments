import tensorflow as tf
import os
from .model import TensorflowDenseNet

MODEL_DIR = "/var/ellie/models/mnist"
MODEL_FILE = os.path.join(MODEL_DIR, "model.ckpt")

SEED = 123
SCOPE = "mnist_softmax_regression"
spec = [
    {"name": "input", "in": 784, "out": 1024, "activation": tf.nn.relu},
    {"name": "hidden1", "in": 1024, "out": 128, "activation": tf.nn.relu},
    {"name": "hidden2", "in": 128, "out": 10, "activation": tf.nn.softmax}]

dense_4_layers = TensorflowDenseNet(SCOPE, SEED, spec)
