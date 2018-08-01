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
     Created on 1st August, 2018 @ 10:47 AM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
# Supress TensorFlow import warnings.
from omniglot import Dataset

import tensorflow as tf

from tensorflow import keras


class Network(object):
    """Light-weight implementation of the SiameseNetwork model."""

    def __init__(self, num_classes=1, **kwargs):
        # Extract Keyword arguments.
        self.num_classes = num_classes
        self._input_shape = kwargs.get('input_shape', (105, 105, 1))
        self._verbose = kwargs.get('verbose', 1)

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
        # Extract keyword arguments.
        lr = kwargs.get('lr', 1e-3)

        # Optimizer.
        optimizer = keras.optimizers.Adam(lr=lr)

        # TODO: Get layerwise learning rates and momentum annealing scheme described in paperworking.
        self._model.compile(loss="binary_crossentropy",
                            optimizer=optimizer, metrics=['accuracy'])

        # Log summary if verbose is 'on'.
        self._log(callback=self._model.summary)

        # Parameter count.
        # n_params = self._model.count_params()
        # self._log(f'Network has {n_params:,} parameters.')

    def train(self, train_data: Dataset, valid_data: Dataset=None, batch_size:int=64, **kwargs):

        # Extract keyword arguments.
        epochs = kwargs.setdefault('epochs', 1)
        steps_per_epoch = kwargs.setdefault('steps_per_epoch', 128)

        # Get batch generators.
        train_gen = train_data.next_batch(batch_size=batch_size)

        # Fit the network.
        if valid_data is None:
            # without validation set.
            self._model.fit_generator(train_gen, **kwargs)
        else:
            valid_gen = valid_data.next_batch(batch_size=batch_size)
            # with validation set.
            self._model.fit_generator(train_gen, validation_data=valid_gen,
                                      validation_steps=batch_size, **kwargs)

    def _log(self, *args, **kwargs):
        # No logging if verbose is not 'on'.
        if self._verbose is not 1:
            return

        # Handle for callbacks.
        callback = kwargs.setdefault('callback', None)
        params = kwargs.setdefault('params', None)

        # Call any callbacks if it is callable.
        if callback and callable(callback):
            # Callback with no params or with params.
            callback() if params is None else callback(params)

        # Remove callback & params keys.
        kwargs.pop('callback')
        kwargs.pop('params')

        # Log other args & kwargs.
        print(*args, **kwargs)


if __name__ == '__main__':
    # Instantiate Siamese Network.
    net = Network()

    # Create some training data.
    train = Dataset(mode=Dataset.Mode.TRAIN)

    # Train the model.
    net.train(train_data=train)
