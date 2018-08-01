"""Utility module for few-shot learning.

   @description
     For converting values to Tensor & pre-paring dataset for `tf.estimator` API.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola/

   @project
     File: utils.py
     Package: omniglot
     Created on 21 Jul, 2018 @ 5:26 AM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""

import numpy as np
import tensorflow as tf

from data import Dataset


__all__ = [
    'Generator',
    'np_input_fn', 'tf_input_fn',
    'to_tensor', 'make_dataset'
]


class Generator(tf.keras.utils.Sequence):
    """Sequence generator wrapper around the `omniglot.Dataset` class."""

    def __init__(self, dataset: Dataset, batch_size: int=128):
        """omniglot.utils.Generator.__init__ now _that_

        Args:
            dataset (omniglot.Dataset): A dataset instance.
            batch_size (int, optional): Defaults to 128. Mini-batch size.
        """

        self._dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        """Length of Generator batch"""
        return int(np.ceil(len(self._dataset) / float(self.batch_size)))

    def __getitem__(self, idx):
        return self._dataset.get(batch_size=self.batch_size)

    @classmethod
    def fromPath(cls, batch_size=128, *args, **kwargs):
        """Create generator from a given path.

        Args:
            batch_size (int, optional): Defaults to 128. Mini-batch_size.
            See `omniglot.Dataset`.

        Keyword Args:
            See `omniglot.Dataset` for options.

        Returns:
            `omniglot.utils.Generator`: A Generator instance.
        """

        # Create a dataset object.
        dataset = Dataset(*args, **kwargs)

        # Instantiate a Generator.
        inst = cls(dataset, batch_size)

        # Return generator instance.
        return inst


def to_tensor(func):
    """Decorator function to (maybe) convert input pairs to tensors.

    Args:
        func (any): Function to be wrapped.

    Returns:
        any: Converted function.
    """

    def converter(*args, **kwargs):
        """Wrapper function.

        Keyword Args:
          ret_type (str, optional): Return type.
            one of 'np', 'numpy' or 'tf', 'tensor'. Defaults to 'np'.
        Returns:
          tuple: Input pairs & targets.
        """
        # Return types.
        ret_types = ('np', 'numpy', 'tf', 'tensor')

        # Default is numpy.
        ret_type = kwargs.get('ret_type', ret_types[0])

        np_pairs, np_targets = func(*args, **kwargs)

        if ret_type.lower() in ret_types[:2]:
            return np_pairs, np_targets
        else:
            # Convert numpy pairs & targets to Tensors.
            pairs = [tf.constant(np_pairs[0], name="input1"),
                     tf.constant(np_pairs[1], name="input2")]
            targets = tf.constant(np_targets, name="target")

        return pairs, targets

    return converter


def make_dataset(features: np.ndarray, labels: np.ndarray=None, **kwargs):
    """Create and transform input data using the TF_DATA API.

    Args:
        features (np.ndarray): Input pairs.
        labels (np.ndarray, optional): Defaults to None. Output labels.

    Returns:
        tf.data.Dataset: TF Dataset object.
    """

    # Extract Keyword arguments.
    batch_size = kwargs.get('batch_size', 64)
    buffer_size = kwargs.get('buffer_size', 500)
    repeat_count = kwargs.get('repeat_count', 10)

    # Create dataset from tensor slices.
    if labels is not None:
        # Using both features and labels.
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    else:
        # Using only input features.
        dataset = tf.data.Dataset.from_tensor_slices(features)

    # Dataset transformation.
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.repeat(count=repeat_count)

    return dataset


def np_input_fn(x: np.ndarray, y: np.ndarray=None, epochs: int=1, **kwargs):
    """NumPy input function.

    Args:
        x (np.ndarray): Input pairs.
        y (np.ndarray, optional): Defaults to None. 1-D target labels.
        epochs (int, optional): Defaults to 1. Number of training epochs.

    Keyword Args:
        See `tf.estimator.inputs.numpy_input_fn`.

    Returns:
        any: Function, that has signature of () -> (dict of `x`, `y`)
    """
    # X feature columns.
    x = {'input_1': x[0], 'input_2': x[1]}

    # Default Keyword arguments.
    kwargs.setdefault('shuffle', False)
    kwargs.setdefault('num_epochs', epochs)

    # Create a numpy input function.
    fn = tf.estimator.inputs.numpy_input_fn(x=x, y=y, **kwargs)

    return fn


def tf_input_fn(x: np.ndarray, y: np.ndarray, **kwargs):
    """TensorFlow input function using the `tf.data` API.

    Args:
        x (np.ndarray or tf.Tensor): Input pairs.
        y (labels): 1-D target labels.

    Keyword Args:
        See `utils.make_dataset`.

    Returns:
        any: TF_ESTIMATOR  input function.
    """
    # For input pairs.
    x = {'input_1': x[0], 'input_2': x[1]}

    return lambda: make_dataset(x, y, **kwargs)


if __name__ == '__main__':
    # Image pairs.
    pair1 = np.random.randn(500, 105, 105, 1)
    pair2 = np.random.randn(500, 105, 105, 1)

    # Dummy dataset.
    pairs = np.array([pair1, pair2], dtype=np.int32)
    labels = np.random.choice(2, size=500)
    print(pairs.shape, labels.shape)

    # Test numpy & tensor input functions.
    print(np_input_fn(pairs, labels))
    print(tf_input_fn(pairs, labels)())
