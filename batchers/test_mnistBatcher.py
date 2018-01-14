from batchers import MnistBatcher


class TestMnistBatcher(object):
    def test_extract_data(self):
        m = MnistBatcher('./testdata/t10k-images-idx3-ubyte.gz', 30)
        assert m.data.shape == (30, 28, 28, 1)
        assert m.data[0, 0, 0, 0] == -0.5

    def test_next_batch_precisely_exhausted(self):
        b = MnistBatcher('./testdata/t10k-images-idx3-ubyte.gz', 30)
        assert b.has_more()
        p = b.next_batch(20)
        assert b.has_more()
        assert p.shape[0] == 20
        p = b.next_batch(10)
        assert not b.has_more()
        assert p.shape[0] == 10
        p = b.next_batch(10)
        assert not b.has_more()
        assert p.shape[0] == 0

    def test_next_batch_over_exhausted(self):
        b = MnistBatcher('./testdata/t10k-images-idx3-ubyte.gz', 30)
        assert b.has_more()
        p = b.next_batch(20)
        assert b.has_more()
        assert p.shape[0] == 20
        p = b.next_batch(15)
        assert not b.has_more()
        assert p.shape[0] == 10
        p = b.next_batch(10)
        assert not b.has_more()
        assert p.shape[0] == 0

    def test_reset(self):
        b = MnistBatcher('./testdata/t10k-images-idx3-ubyte.gz', 30)
        p = b.next_batch(30)
        assert not b.has_more()
        b.reset()
        assert b.has_more()
        p1 = b.next_batch(30)
        assert p[5, 14, 14] == p1[5, 14, 14]
