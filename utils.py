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

# from functools import wraps


def to_tensor(func):
    """Decorator function to convert input pairs to tensors.

    Args:
        func (any): Function to be wrapped.

    Returns:
        any: Converted function.
    """

    # @wraps
    def converter(*args, **kwargs):
        """Wrapper function

        Returns:
          tuple: Input pairs & targets.
        """

        ret_type = kwargs.get('ret_type', 'np')

        np_pairs, np_targets = func(*args, **kwargs)

        # Convert numpy pairs & targets to Tensors.
        pairs = [tf.constant(np_pairs[0], name="input1"),
                 tf.constant(np_pairs[1], name="input2")]
        targets = tf.constant(np_targets, name="target")

        return pairs, targets

    return converter
