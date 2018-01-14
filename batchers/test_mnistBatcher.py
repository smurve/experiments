from batchers import MnistBatcher


class TestMnistBatcher(object):
    def test_extract_data(self):
        m = MnistBatcher('./testdata/t10k-images-idx3-ubyte.gz', 30)
        assert m.data.shape == (30, 28, 28, 1)
        assert m.data[0, 0, 0, 0] == -0.5
