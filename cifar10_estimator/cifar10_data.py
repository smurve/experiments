import tensorflow as tf
import cifar10


def input_fn(data_dir,
             subset,
             num_shards,
             batch_size,
             use_distortion_for_training=True):
    """Create input graph for model.

    Args:
      data_dir: Directory where TFRecords representing the dataset are located.
      subset: one of 'train', 'validate' and 'eval'.
      num_shards: num of towers participating in data-parallel training.
      batch_size: total batch size for training to be divided by the number of
      shards.
      use_distortion_for_training: True to use distortions.
    Returns:
      two lists of tensors for features and labels, each of num_shards length.
    """
    with tf.device('/cpu:0'):
        use_distortion = subset == 'train' and use_distortion_for_training
        dataset = cifar10.Cifar10DataSet(data_dir, subset, use_distortion)
        image_batch, label_batch = dataset.make_batch(batch_size)
        if num_shards <= 1:
            # No GPU available or only 1 GPU.
            return [image_batch], [label_batch]

        # Note that passing num=batch_size is safe here, even though
        # dataset.batch(batch_size) can, in some cases, return fewer than batch_size
        # examples. This is because it does so only when repeating for a limited
        # number of epochs, but our dataset repeats forever.
        image_batch = tf.unstack(image_batch, num=batch_size, axis=0)
        label_batch = tf.unstack(label_batch, num=batch_size, axis=0)
        feature_shards = [[] for _ in range(num_shards)]
        label_shards = [[] for _ in range(num_shards)]
        for i in range(batch_size):
            idx = i % num_shards
            feature_shards[idx].append(image_batch[i])
            label_shards[idx].append(label_batch[i])
        feature_shards = [tf.parallel_stack(x) for x in feature_shards]
        label_shards = [tf.parallel_stack(x) for x in label_shards]
        return feature_shards, label_shards
