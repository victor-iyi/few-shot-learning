"""Utility module for loading dataset..

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola/

   @project
     File: utils.py
     Created on 18 May, 2018 @ 5:26 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
import tensorflow as tf

from functools import wraps


def to_tensor(func, *args, **kwargs):
    """Decorator function to convert input pairs to tensors.

    Args:
        func (any): Function to be wrapped.

    Returns:
        any: Converted function.
    """

    @wraps
    def converter():
        np_pairs, targets = func(*args, **kwargs)

        # Convert numpyb pairs to Tensors.
        pairs = [tf.constant(np_pairs[0]), tf.constant(np_pairs[1])]

        return pairs, targets

    return converter
