"""Utility module for few-shot learning.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola/

   @project
     File: utils.py
     Created on 21 Jul, 2018 @ 5:26 AM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
import tensorflow as tf


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


def np_input_fn(x, y=None, epochs=1, **kwargs):
    """NumPy input function.

    Args:
        x (np.ndarray): Input pairs.
        y (np.ndarray, optional): Defaults to None. 1-D target labels.
        epochs (int, optional): Defaults to 1. Number of training epochs.

    Returns:
        any: Function, that has signature of () -> (dict of `x`, `y`)
    """
    # X feature columns.
    x = {'input_1': x[0], 'input_2': x[1]}

    # Create a numpy input function.
    fn = tf.estimator.inputs.numpy_input_fn(x=x, y=y,
                                            num_epochs=epochs,
                                            **kwargs)
    return fn


def tf_input_fn(x, y, **kwargs):
    """TensorFlow input function using the `tf.data` API.

    Args:
        x (np.ndarray or tf.Tensor): Input pairs.
        y (labels): 1-D target labels.

    Returns:
        any: TF_ESTIMATOR  input function.
    """

    return lambda: make_dataset(x, y, **kwargs)


def make_dataset(features, labels=None, **kwargs):
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
