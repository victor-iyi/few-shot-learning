"""Base class for building models for One-shot learning.

   @description
     For benchmarking other model's accuracy. An ideal model should perform
     better than this.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola/

   @project
     File: benchmark.py
     Package: omniglot
     Created on 3rd August, 2018 @ 08:11 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
import numpy as np


class Benchmark(object):

    def __init__(self, data, **kwargs):
        self.data = data

    def predict(self, pairs, targets):
        """Perform a nearest-neighbor lookup for a one-shot task.

        Args:
            pairs (tuple): Containing image pairs each of dimension
                (width, height, channel).
            targets (np.ndarray): array-like - 0 if pairs are of the same class,
                0 otherwise.

        Returns:
            int: Prediction based on nearest neighbor lookup. Outputs 1 if
                predicted that image pairs are of the same class, 0 otherwise.
        """
        # To store distance summed over each neighbor lookup.
        distance = np.zeros_like(targets)

        for i, pair in enumerate(pairs):
            dist = np.sqrt(pair[0] ** 2 - pair[1] ** 2)
            distance[i] = np.sum(dist)

        # The closest distance wins is the same as correct target.
        correct = np.argmin(distance) == np.argmax(targets)

        return 1 if correct else 0

    def score(self, n: int, trials: int=None, **kwargs):
        """Evaluate accuracy of nearest neighbor lookup on N-way one-shot task
        for `trials` number of trials.

        Args:
            n (int): N-way one-shot task. Size of neighbors to compare.
                Note: `n=1` for one-shot learning task, but varies for few-shot
                learning.
            trials (int, optional): Defaults to half of `self.data`. How many
            trails to be predicted.

        Keyword Args:
            verbose (int, optional): Defaults to 1. Values are either 0 or 1.
        """
        # Extract keyword arguments.
        verbose = kwargs.get('verbose', 1)

        trials = trials or len(self.data) // 2
        correct = 0

        for i in range(trials):
            pairs, targets = self.data.one_shot_task(n)
            correct += self.predict(pairs, targets)
        if verbose:
            print(f'{n:,}-way few-shot task w/ {trials:,} trials = {correct/trials:.2%}')

        return correct / trials
