from .batcher import MnistBatcher
from .batcher import img_test, img_train, lbl_test, lbl_train

train_batcher = MnistBatcher(img_file=img_train, lbl_file=lbl_train, num_samples=60000)
test_batcher = MnistBatcher(img_file=img_test, lbl_file=lbl_test, num_samples=10000)
