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
from scipy.spatial.distance import euclidean


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

        for i, (pair1, pair2) in enumerate(pairs):
            distance[i] = np.sum(euclidean(pair1, pair2))

        # The closest distance wins is the same as correct target.
        correct = np.argmin(distance) == np.argmax(targets)

        return 1 if correct else 0

    def score(self, n: int, trials: int=None):
        """Evaluate accuracy of nearest neighbor lookup on N-way one-shot task
        for `trials` number of trials.

        Args:
            n (int): N-way one-shot task. Size of neighbors to compare.
                Note: `n=1` for one-shot learning task, but varies for few-shot
                learning.
            trials (int, optional): Defaults to half of `self.data`. How many
            trails to be predicted.
        """
        trails = trials or len(self.data) // 2
        correct = 0

        for i in range(trails):
            pairs, targets = self.data.one_shot_task(n)
            correct += self.predict(pairs, targets)

        return correct / trials


def test_nn_accuracy(N_ways, n_trials, loader):
    """Returns accuracy of one shot."""
    print(("Evaluating nearest neighbour on {} unique {}"
           "way one-shot learning tasks ...").format(n_trials, N_ways))

    n_right = 0

    for i in range(n_trials):
        pairs, targets = loader.make_oneshot_task(N_ways, "val")
        correct = nearest_neighbour_correct(pairs, targets)
        n_right += correct

    return 100.0 * n_right / n_trials
