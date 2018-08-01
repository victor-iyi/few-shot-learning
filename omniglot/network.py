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

        self.num_classes = num_classes

        # Extract Keyword arguments.
        self._input_shape = kwargs.get('input_shape', (105, 105, 1))
        self._verbose = kwargs.get('verbose', 1)
        self._model_dir = kwargs.get('model_dir', 'saved/models/').rstrip('/')

        # Create model directory if it doesn't already exit.
        if not tf.gfile.IsDirectory(self._model_dir):
            tf.gfile.MakeDirs(self._model_dir)

        # Path to save model's weights.
        self._model_weights = f'{self._model_dir}/network-weights.h5'

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

    def train(self, train_data: Dataset, valid_data: Dataset=None, batch_size: int=64, **kwargs):

        # Extract keyword arguments.
        epochs = kwargs.setdefault('epochs', 1)
        steps_per_epoch = kwargs.setdefault('steps_per_epoch', 128)

        kwargs.setdefault('verbose', self._verbose)

        # Get batch generators.
        train_gen = train_data.next_batch(batch_size=batch_size)
        try:
            # Fit the network.
            if valid_data is None:
                # without validation set.
                self._model.fit_generator(train_gen, **kwargs)
            else:
                valid_gen = valid_data.next_batch(batch_size=batch_size)
                # with validation set.
                self._model.fit_generator(train_gen, validation_data=valid_gen,
                                        validation_steps=batch_size, **kwargs)
        except KeyboardInterrupt:
            # When training is unexpectedly stopped!
            self._log('\nTraining interrupted by user!')

            # Save learned weights.
            self.save_weights()

    def save_weights(self):
        """Save Model's weights to an h5 file."""

        # Pretty prints.
        self._log(f'\n{"-" * 65}\nSaving model...')
    
        # Save model weights.
        self._model.save_weights(filepath=self._model_weights,
                                 overwrite=True, save_format=None)

        # Pretty prints.
        self._log(f'Saved model weights to "{self._save_path}"!\n{"-" * 65}\n')

    def _log(self, *args, **kwargs):
        """Logging method helper based on verbosity."""

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

    @staticmethod
    def triplet_loss(y_true, y_pred, alpha=0.2):
        """Triplet Loss function to compare pairs of

        Args:
            y_pred (tf.Tensor): Encoding of anchor & positive example.
            y_true (tf.Tensor): Encoding of anchor & negative example.
            alpha (float, optional): Defaults to 0.2. Margin added to f(A, P).

        Returns:
            tf.Tensor: Triplet loss.
        """

        # Triplet loss for a single image.
        loss = tf.maximum(y_true - y_pred + alpha, 0)

        # Sum over all images.
        return tf.reduce_sum(loss, axis=1, name="Triplet_Loss")

    @staticmethod
    def binary_crossentropy(y_true, y_pred):
        """Binary crossentropy between an output tensor and a target tensor.

        Args:
            y_true: A tensor with the same shape as `output`.
            y_pred: A tensor.

        Returns:
            tf.tensor: Binary crossentropy loss.
        """

        # Binary crossentropy loss function.
        return keras.losses.binary_crossentropy(y_true, y_pred)

    @staticmethod
    def contrastive_loss(y_true, y_pred, alpha=0.2):
        """Contrastive loss function.

        Binary cross entropy between the predictions and targets.
        There is also a L2 weight decay term in the loss to encourage
        the network to learn smaller/less noisy weights and possibly
        improve generalization:

        L(x1, x2, t) = t⋅log(p(x1 ∘ x2)) + (1−t)⋅log(1 − p(x1 ∘ x2)) + λ⋅||w||2

        Args:
            y_pred (any): Predicted distance between two inputs.
            y_true (any): Ground truth or target, t (where, t = [1 or 0]).

            alpha (float, optional): Defaults to 0.2. Slight margin
                added to prediction to avoid 0-learning.

        Returns:
            tf.Tensor: Constrictive loss function.
        """

        loss = y_true * tf.log(y_true) + (1 - y_pred) * tf.log(1 - y_pred) + alpha

        return tf.reduce_mean(loss, name="contrastive_loss")

    @property
    def model(self):
        """Network background model.
        
        Returns:
            keras.Model: Underlaying model used by Network.
        """

        return self._model

    @property
    def weight_path(self):
        """Model saved weight path.
        
        Returns:
            str: Path to an h5 file.
        """

        return self._model_weights
