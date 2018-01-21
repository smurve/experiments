from .batcher import CIFAR10Batcher


def test_single_file():

    batcher = CIFAR10Batcher(["data_batch_3"])
    imgs, lbls = batcher.next_batch(5000)
    assert batcher.has_more() is True
    assert len(imgs) == 5000
    assert len(lbls) == 5000

    imgs, lbls = batcher.next_batch(5000)
    assert len(imgs) == 5000
    assert len(lbls) == 5000

    assert batcher.has_more() is False

    batcher.reset()
    assert batcher.has_more() is True

    batcher.next_batch(5000)
    imgs, _ = batcher.next_batch(6000)
    assert len(imgs) == 5000
    assert batcher.has_more() is False


def test_two_files():

    batcher = CIFAR10Batcher(["data_batch_3", "data_batch_4"])
    imgs, lbls = batcher.next_batch(5000)
    assert batcher.has_more() is True
    assert len(imgs) == 5000
    assert len(lbls) == 5000

    imgs, lbls = batcher.next_batch(5000)
    assert len(imgs) == 5000
    assert len(lbls) == 5000

    assert batcher.has_more() is True

    batcher.next_batch(5000)
    imgs, _ = batcher.next_batch(6000)
    assert len(imgs) == 5000
    assert batcher.has_more() is False

    batcher.reset()

    batcher.next_batch(8000)
    imgs, _ = batcher.next_batch(4000)
    assert len(imgs) == 4000
    assert batcher.has_more() is True

    imgs, _ = batcher.next_batch(8000)
    assert len(imgs) == 8000
    assert batcher.has_more() is False

