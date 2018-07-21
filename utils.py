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
