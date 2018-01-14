from .mnist import MnistBatcher


def mnist_batcher(filename, num_images):
    return MnistBatcher(filename, num_images)
