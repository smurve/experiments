import argparse


class MNISTArgParser(argparse.ArgumentParser):

    def __init__(self):
        super(MNISTArgParser, self).__init__()

        self.add_argument(
            '--multi_gpu', action='store_true',
            help='If set, run across all available GPUs.')
        self.add_argument(
            '--batch_size',
            type=int,
            default=100,
            help='Number of images to process in a batch')
        self.add_argument(
            '--data_dir',
            type=str,
            default='/tmp/mnist_data',
            help='Path to directory containing the MNIST dataset')

        self.add_argument(
            '--task_index',
            type=int,
            default=0,
            help='The index (starting with 0) of this task')
        self.add_argument(
            '--job_name',
            type=str,
            default='worker',
            help="Either 'ps' for parameter server or 'worker'")

        self.add_argument(
            '--model_dir',
            type=str,
            default='/tmp/mnist_model',
            help='The directory where the model will be stored.')
        self.add_argument(
            '--train_epochs',
            type=int,
            default=1,
            help='Number of epochs to train.')
        self.add_argument(
            '--data_format',
            type=str,
            default=None,
            choices=['channels_first', 'channels_last'],
            help='A flag to override the data format used in the model. '
                 'channels_first provides a performance boost on GPU but is not always '
                 'compatible with CPU. If left unspecified, the data format will be '
                 'chosen automatically based on whether TensorFlow was built for CPU or '
                 'GPU.')
        self.add_argument(
            '--export_dir',
            type=str,
            help='The directory where the exported SavedModel will be stored.')
