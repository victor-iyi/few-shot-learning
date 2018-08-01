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
     Created on 1 August, 2018 @ 10:47 AM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""

import tensorflow as tf

from tensorflow import keras


class Network(object):
    """Light-weight implementation of the SiameseNetwork model."""

    def __init__(self, num_classes=1, **kwargs):
        # Extract Keyword arguments.
        self.num_classes = num_classes
        self._input_shape = kwargs.get('input_shape', (500, 500, 1))

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
        distance_layer = keras.layers.Lambda(lambda x: tf.abs(x[0] - x[1]))

        # Call this layer on list of two input tensors.
        distance = distance_layer([encoder_1st, encoder_2nd])

        # Model prediction: if image pairs are of same letter.
        output_layer = keras.layers.Dense(num_classes, activation='sigmoid')
        outputs = output_layer(distance)

        # Model.
        self._model = keras.Model(inputs=[pair_1st, pair_2nd], outputs=outputs)

    def train(self, *args, **kwargs):
        # Extract keyword arguments.
        lr = kwargs.get('lr', 1e-3)

        # Optimizer.
        optimizer = keras.optimizers.Adam(lr=lr)

        # TODO: Get layerwise learning rates and momentum annealing scheme described in paperworking.
        self._model.compile(loss="binary_crossentropy",
                            optimizer=optimizer)
