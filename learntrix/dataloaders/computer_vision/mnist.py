from collections.abc import Iterator

import tensorflow_datasets as tfds

from learntrix.types import Batch


def load_mnist_dataset(
    split: str,
    *,
    shuffle: bool,
    batch_size: int,
) -> Iterator[Batch]:
    """
    Loads the MNIST dataset

    Args:
        split (str): Either 'train' or 'test'
        shuffle (bool): Whether to shuffle the data
        batch_size (int): Size of the batch

    Returns:
        Iterator[Batch]: Iterator over the MNIST dataset
    """
    ds, ds_info = tfds.load("mnist:3.*.*", split=split, with_info=True)
    ds.cache()
    if shuffle:
        ds = ds.shuffle(ds_info.splits[split].num_examples, seed=0)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x: Batch(**x))
    return iter(tfds.as_numpy(ds))
