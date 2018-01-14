import gzip
import numpy


class MnistBatcher:

    """
    A mini-batch provider for MNIST Images
    """

    def __init__(self, filename, num_images):
        self.IMAGE_SIZE = 28
        self.PIXEL_DEPTH = 255
        self.filename = filename
        self.num_images = num_images
        self.current_image = 0
        self.data = self.extract_data(filename, num_images)

    def extract_data(self, filename, num_images):
        with gzip.open(filename) as bytestream:
            bytestream.read(16)

            buf = bytestream.read(self.IMAGE_SIZE * self.IMAGE_SIZE * num_images)
            data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
            data = (data - (self.PIXEL_DEPTH / 2.0)) / self.PIXEL_DEPTH
            data = data.reshape(num_images, self.IMAGE_SIZE, self.IMAGE_SIZE, 1)
            return data

    def next_batch(self, batch_size):
        last_index = min(self.num_images, self.current_image + batch_size)
        batch = self.data[self.current_image:last_index, :, :, 0]
        self.current_image = last_index
        return batch

    def has_more(self):
        return self.current_image < self.num_images

    def reset(self):
        self.current_image = 0
