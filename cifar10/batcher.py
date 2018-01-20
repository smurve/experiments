import pickle
import os
import numpy as np

DATA_DIR = "/var/ellie/data/cifar10"
TRAINING = list(['data_batch_%s' % (n+1) for n in range(4)])


def image(data):
    """
    This is the correct way to extract an image from the raw data for display
    """
    return data.T.reshape(3, 32, 32).transpose([1, 2, 0])


class CIFAR10Batcher:
    """
    A mini-batch provider for CIFAR10 Images
    """

    def __init__(self, files=TRAINING):
        """
        Initialize by caching the first file, keeps only a single file in memory at any point in time
        :param files:
        Array of file names to use as an epoch. Defaults to all of the training files
        """
        self.IMAGE_SIZE = 32
        self.PIXEL_DEPTH = 255
        self.samples_per_file = 10000
        self.files = files
        self.num_samples = self.samples_per_file * len(self.files)
        self.current_image = 0
        self.current_file = 0
        self.preprocessor = None
        self.imgs, self.lbls = self.extract_data(files[0])

    def extract_data(self, filename):
        path = os.path.join(DATA_DIR, filename)
        dictionary = self.unpickle(path)
        imgs = dictionary[b'data']
        lbls = dictionary[b'labels']
        return imgs, lbls

    @staticmethod
    def unpickle(file):
        with open(file, 'rb') as fo:
            dictionay = pickle.load(fo, encoding='bytes')
        return dictionay

    def pos_for(self, index):
        return int(index / self.samples_per_file), index % self.samples_per_file

    def next_batch(self, batch_size):
        """
        next batch of shape [batch_size, 3072]
        :param batch_size: the number of samples required.
        :return:
        """

        last_index = min(self.num_samples, self.current_image + batch_size)

        file0, pos0 = self.pos_for(self.current_image)
        file1, pos1 = self.pos_for(last_index)

        if file0 == file1:
            imgs = self.imgs[pos0:pos1, :]
            lbls = self.lbls[pos0:pos1]
        else:
            imgs = self.imgs[pos0:, :]
            lbls = self.lbls[pos0:]
            self.current_file += 1
            if self.current_file < len(self.files):
                self.imgs, self.lbls = self.extract_data(self.files[self.current_file])
                diff = batch_size - len(imgs)
                if diff > 0:
                    imgs1 = self.imgs[0: diff]
                    lbls1 = self.lbls[0: diff]
                    imgs = np.vstack((imgs, imgs1))
                    lbls = np.vstack((lbls, lbls1))

        self.current_image = self.current_image + len(imgs)

        if self.preprocessor is not None:
            imgs = self.preprocessor(imgs)
        return imgs, lbls

    #
    #
    #
    def has_more(self):
        return self.current_image < self.num_samples

    def reset(self):
        self.current_image = 0
        self.current_file = 0

    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor
