import os
import png
import numpy as np
import tensorflow as tf
from models.particular import dense_4_layers
from models.particular import MODEL_FILE
from mnist import test_batcher as batcher


class MnistPredictionService:

    def __init__(self, img_dir='app/images'):
        self.img_dir = img_dir

    def clean_img_dir(self):
        for img in os.listdir(self.img_dir):
            path = os.path.join(self.img_dir, img)
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
            failed = (1 - np.equal(predicted, lbls)) * range(batch_size)
            failed_indices = [i for i in filter(lambda x: x > 0, failed)]
            failed_records = [{"index": idx, "lbl": lbls[idx], "pred": predicted[idx]}
                              for idx in failed_indices]
            return failed_records

    @staticmethod
    def save_png(img, name):
        img = img.reshape(28, 28)
        img = np.int32((img + 0.5) * 255)
        with open("%s.png" % name, "wb") as h:
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
        for record in failed_records[0:min(max_nr_files, len(failed_records)+1)]:
            index = record["index"]
            img = imgs[index]
            name = "%s-%s-%s" % (index, record["lbl"], record["pred"])
            self.save_png(img, os.path.join(self.img_dir, name))
