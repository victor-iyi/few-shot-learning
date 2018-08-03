"""Omniglot helper package.

   @description
     Network - For training, validating/evaluating & predictions with SiameseNetwork.
     Dataset - For pre-processing and loading the Omniglot dataset.
     Visualize - For visualizing the Omniglot dataset & model.
     utils - For converting values to Tensor & pre-paring dataset for `tf.estimator` API.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola/

   @project
     File: __init__.py
     Package: omniglot
     Created on 18 May, 2018 @ 5:22 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""

__author__ = 'Victor I. Afolabi'

# Supress TensorFlow import warnings.
from omniglot import supress
from omniglot import utils

from omniglot.data import Dataset
from omniglot.data import n_runs, compressed_dir, data_dir, base_dir
from omniglot.visualize import Visualize

from omniglot.benchmark import Benchmark

from omniglot.base import BaseNetwork, Loss
from omniglot.network import SiameseNetwork, EncoderNetwork


__all__ = [
    # Dataset helpers.
    'base_dir', 'data_dir',
    'compressed_dir', 'n_runs',
    'Dataset',

    # Visualization helper.
    'Visualize',

    # Benchmarking helper.
    'Benchmark',

    # Network/Model helpers.
    'SiameseNetwork',
    'EncoderNetwork',
    'BaseNetwork', 'Loss',

    # Input helpers.
    'utils',
]
