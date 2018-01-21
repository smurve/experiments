import os
import png
import numpy as np
import tensorflow as tf
from models.particular_mnist import dense_4_layers
from models.particular_mnist import MODEL_FILE
from mnist import test_batcher as batcher


class FailedPrediction:
    def __init__(self, idx, file, lbl, pred):
        self.idx = idx
        self.file = file
        self.lbl = lbl
        self.pred = pred

    def __str__(self):
        return "[%s] File %s, true: %s - pred: %s" % (self.idx, self.file, self.lbl, self.pred)


class MnistPredictionService:

    def __init__(self, img_dir='images'):
        self.img_dir = img_dir
        if not os.path.isdir(self.app_img_dir()):
            os.mkdir("images")

    def app_img_dir(self):
        return os.path.join("app", self.img_dir)

    def clean_img_dir(self):
        for img in os.listdir(self.app_img_dir()):
            path = os.path.join(self.app_img_dir(), img)
            try:
                if os.path.isfile(path):
                    os.unlink(path)
            except Exception as e:
                print(e)

    @staticmethod
    def list_failed(batch_size=10000):
        batch_size = min(batch_size, 10000)
        tf.reset_default_graph()
        with tf.Session() as sess:
            model = dense_4_layers()
            saver = tf.train.Saver()
            saver.restore(sess, MODEL_FILE)
            batcher.reset()
            img, lbls = batcher.next_batch(batch_size)
            img = img.reshape(-1, 784)
            lbls = np.int32(lbls)
            inference = sess.run(model.activation, feed_dict={model.samples: img})
            predicted = np.argmax(inference, axis=1)
            failed = np.multiply((1 - np.equal(predicted, lbls)), range(batch_size))
            failed_indices = [i for i in filter(lambda x: x > 0, failed)]
            failed_records = [{"index": idx, "lbl": lbls[idx], "pred": predicted[idx]}
                              for idx in failed_indices]
            return failed_records

    @staticmethod
    def save_png(img, pngfile):
        img = img.reshape(28, 28)
        img = np.int32((img + 0.5) * 255)
        with open(pngfile, "wb") as h:
            w = png.Writer(28, 28, greyscale=True)
            w.write(h, img)

    def create_failed_pngs(self, search_size, max_nr_files=300):
        """
        :param max_nr_files: max number of files to create
        :param search_size: search at max this number of images for failed predictions
        """
        self.clean_img_dir()
        batcher.reset()
        imgs, _ = batcher.next_batch(search_size)
        failed_records = self.list_failed(search_size)
        all_img = []
        for record in failed_records[0:min(max_nr_files, len(failed_records)+1)]:
            index = record["index"]
            img = imgs[index]
            name = "%s-%s-%s.png" % (index, record["lbl"], record["pred"])
            self.save_png(img, os.path.join(self.app_img_dir(), name))
            name = os.path.join(self.img_dir, name)
            all_img.append(FailedPrediction(index, name, record["lbl"], record["pred"]))
        return all_img
