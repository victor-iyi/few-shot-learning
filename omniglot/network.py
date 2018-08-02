"""Helper class for building Siamese model for One-shot learning.

   @description
     For training, validating/evaluating & predictions with SiameseNetwork.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola/

   @project
     File: network.py
     Package: omniglot
     Created on 13 July, 2018 @ 9:10 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""

import tensorflow as tf
from tensorflow import keras

from omniglot import BaseNetwork, Dataset


class EncoderNetwork(BaseNetwork):
    """Implementation of an encoder SiameseNetwork model."""

    def __init__(self, num_classes=1, **kwargs):
        super(EncoderNetwork, self).__init__(**kwargs)

    def build(self,  **kwargs):
        # # Number of output classes.
        # num_classes = kwargs.get('num_classes', 1)
        dropout = kwargs.get('dropout', 0.2)

        # Input pair inputs.
        pair_1st = keras.Input(shape=self._input_shape)
        pair_2nd = keras.Input(shape=self._input_shape)

        # Siamese Model.
        net = keras.models.Sequential()

        # 1st layer (64@10x10)
        net.add(keras.layers.Conv2D(filters=64, kernel_size=(10, 10),
                                    input_shape=self._input_shape,
                                    activation='relu'))
        net.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

        # 2nd layer (128@7x7)
        net.add(keras.layers.Conv2D(filters=128, kernel_size=(7, 7),
                                    activation='relu'))
        net.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

        # 3rd layer (128@4x4)
        net.add(keras.layers.Conv2D(filters=128, kernel_size=(4, 4),
                                    activation='relu'))
        net.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

        # 4th layer (265@4x4)
        net.add(keras.layers.Conv2D(filters=256, kernel_size=(4, 4),
                                    activation='relu'))
        net.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

        # 5th layer  (9216x4096)
        net.add(keras.layers.Flatten())
        net.add(keras.layers.Dense(units=4096, activation='sigmoid'))

        # Call the Sequential model on each input tensors with shared params.
        encoder_1st = net(pair_1st)
        encoder_2nd = net(pair_2nd)

        # Layer to merge two encoded inputs with the l1 distance between them.
        distance_layer = keras.layers.Lambda(self.dist_func)

        # Call this layer on list of two input tensors.
        distance = distance_layer([encoder_1st, encoder_2nd])

        # Model prediction: if image pairs are of same letter.
        output_layer = keras.layers.Dense(self.num_classes, activation='sigmoid')
        outputs = output_layer(distance)

        # Return a keras Model architecture.
        return keras.Model(inputs=[pair_1st, pair_2nd], outputs=outputs)

    def call(self, inputs, **kwargs):
        """Calls the model on new inputs.

        In this case `call` just reapplies all ops in the graph to the new inputs
        (e.g. build a new computational graph from the provided inputs).

        Args:
            inputs: A tensor or list of tensors.

        Keyword Args:
            training: Boolean or boolean scalar tensor, indicating whether to run
            the `Network` in training mode or inference mode.
            mask: A mask or list of masks. A mask can be
                either a tensor or None (no mask).

        Returns:
            A tensor if there is a single output, or
            a list of tensors if there are more than one outputs.
        """
        return self._model(inputs, **kwargs)


class SiameseNetwork(BaseNetwork):
    """Siamese Neural network for few shot learning."""

    # noinspection SpellCheckingInspection
    def __init__(self, num_classes: int = 1, **kwargs):
        """Implementation of "__Siamese Network__" with parameter specifications as proposed in [this paper](http://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) by Gregory Koch, Richard Zemel and Ruslan Salakhutdinov.

        Args:
            num_classes (int, optional): Defaults to 1. Number of output classes
                in the last layer (prediction layer).

        Keyword Args:
            input_shape (tuple, optional): Defaults to (105, 105, 1). Input shape
                for a single image. Shape in the form: `(width, height, channel)`.
        """

        super(SiameseNetwork, self).__init__(**kwargs)

    def build(self, **kwargs):
        # Optional Keyword arguments.
        num_classes = kwargs.get('num_classes', 1)
        dropout = kwargs.get('dropout', 0.2)

        # Build a sequential model.
        model = keras.models.Sequential()

        # Re-use pooling layer accross feature extraction layers.
        self.pool = keras.layers.MaxPool2D(pool_size=(2, 2))

        # 1st layer (64@10x10)
        self.conv1 = keras.layers.Conv2D(filters=64, kernel_size=(10, 10),
                                         input_shape=self._input_shape,
                                         activation='relu')

        # 2nd layer (128@7x7)
        self.conv2 = keras.layers.Conv2D(filters=128, kernel_size=(7, 7),
                                         activation='relu')

        # 3rd layer (128@4x4)
        self.conv3 = keras.layers.Conv2D(filters=128, kernel_size=(4, 4),
                                         activation='relu')

        # 4th layer (265@4x4)
        self.conv4 = keras.layers.Conv2D(filters=256, kernel_size=(4, 4),
                                         activation='relu')

        # 5th layer  (9216x4096)
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(units=4096, activation='sigmoid')

        # 6th - L1 layer -distance layer.
        self.dist_layer = keras.layers.Lambda(self.dist_func)

        # Output layer (4096x1)
        self.pred = keras.layers.Dense(units=num_classes, activation='sigmoid')

        # Input layer (2, 105, 105, 1)
        input_1 = self._construct()
        input_2 = self._construct()

        model = keras.Model(inputs=input_1, outputs=self.pred)

    def call(self, inputs, **kwargs):
        """Calls the model on new inputs.

        In this case `call` just reapplies all ops in the graph to the new inputs
        (e.g. build a new computational graph from the provided inputs).

        Args:
            inputs: A tensor or list of tensors.

        Keyword Args:
            training: Boolean or boolean scalar tensor, indicating whether to run
            the `Network` in training mode or inference mode.
            mask: A mask or list of masks. A mask can be
                either a tensor or None (no mask).

        Returns:
            A tensor if there is a single output, or
            a list of tensors if there are more than one outputs.
        """

        # Sister networks.
        first = self.__encoder(inputs[0])
        second = self.__encoder(inputs[1])

        # L1 distance.
        distance = self.distance((first, second))

        # Prediction.
        pred = self.prediction(distance)

        # Returns distance and prediction if not in training mode.
        # return pred if training else distance, pred
        return pred


if __name__ == '__main__':
    import numpy as np

    net = SiameseNetwork(loss=SiameseNetwork.triplet_loss)

    # Image pairs in `np.ndarray`.
    first = np.random.randn(1, 105, 105, 1)
    second = np.random.randn(1, 105, 105, 1)

    # Converted to `tf.Tensor`.
    pairs = [tf.constant(first), tf.constant(second)]

    net(pairs)
