from .batcher import MnistBatcher


class TestMnistBatcher(object):

    IMG_FILE = './testdata/t10k-images-idx3-ubyte.gz'
    LBL_FILE = './testdata/t10k-labels-idx1-ubyte.gz'

    def test_extract_data(self):
        b = MnistBatcher(img_file=self.IMG_FILE, lbl_file=self.LBL_FILE, num_samples=30)
        assert b.imgs.shape == (30, 28, 28, 1)
        assert b.imgs[0, 0, 0, 0] == -0.5

    def test_next_batch_precisely_exhausted(self):
        b = MnistBatcher(img_file=self.IMG_FILE, lbl_file=self.LBL_FILE, num_samples=30)
        assert b.has_more()
        p, _ = b.next_batch(20)
        assert b.has_more()
        assert p.shape[0] == 20
        p, _ = b.next_batch(10)
        assert not b.has_more()
        assert p.shape[0] == 10
        p, _ = b.next_batch(10)
        assert not b.has_more()
        assert p.shape[0] == 0

    def test_next_batch_over_exhausted(self):
        b = MnistBatcher(img_file=self.IMG_FILE, lbl_file=self.LBL_FILE, num_samples=30)
        b.next_batch(20)
        p, _ = b.next_batch(15)
        assert not b.has_more()
        assert p.shape[0] == 10
        p, _ = b.next_batch(10)
        assert not b.has_more()
        assert p.shape[0] == 0

    def test_reset(self):
        b = MnistBatcher(img_file=self.IMG_FILE, lbl_file=self.LBL_FILE, num_samples=30)
        p, _ = b.next_batch(30)
        assert not b.has_more()
        b.reset()
        assert b.has_more()
        p1, _ = b.next_batch(30)
        assert p[5, 14, 14] == p1[5, 14, 14]
