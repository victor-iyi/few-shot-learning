"""Base class for building Siamese model for One-shot learning.

   @description
     For training, validating/evaluating & predictions with SiameseNetwork.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola/

   @project
     File: base.py
     Package: omniglot
     Created on 2nd August, 2018 @ 02:23 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
from omniglot import Dataset

from abc import ABCMeta, ababstractmethod, abstabstractproperty

import tensorflow as tf
from tensorflow import keras


class BaseNetwork(object):
    # Abstract Base Class.
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        # Extract Keyword arguments.
        lr = kwargs.get('lr', 1e-3)
        self._verbose = kwargs.get('verbose', 1)
        metrics = kwargs.get('metrics', ['accuracy'])
        self._input_shape = kwargs.get('input_shape', (105, 105, 1))
        self._save_weights_only = kwargs.get('save_weights_only', False)
        optimizer = kwargs.get('optimizer', keras.optimizers.Adam(lr=lr))
        loss_func = kwargs.get('loss_func', BaseNetwork.binary_crossentropy)
        self._model_dir = kwargs.get('model_dir', 'saved/models/').rstrip('/')

        # Create model directory if it doesn't already exit.
        if not tf.gfile.IsDirectory(self._model_dir):
            tf.gfile.MakeDirs(self._model_dir)

        # Path to save model's weights.
        save_path = 'weights.h5' if self._save_weights_only else 'network.h5'
        self._save_path = f'{self._model_dir}/{save_path}'

        # Instantiate a Sequential Model. [Can be overriden].
        self._model = self.build(**kwargs)

        # TODO: Get layerwise learning rates and momentum annealing scheme described in the paper.
        self._model.compile(loss=loss_func, optimizer=optimizer, metrics=metrics)

        # Log summary if verbose is 'on'.
        self._log(callback=self._model.summary)

        # Parameter count.
        # n_params = self._model.count_params()
        # self._log(f'Network has {n_params:,} parameters.')

    def __call__(self, inputs, **kwargs):
        return self._model(inputs, **kwargs)

    @abstractmethod
    def build(self, **kwargs):
        raise NotImplementedError('Sub-class must override `create_model`.')

    @abstractmethod
    def train(self, train_data: Dataset, valid_data: Dataset=None,
              batch_size: int=64, resume_training=True, **kwargs):
        raise NotImplementedError('Sub-class must override `train`.')

    def callbacks(self, **kwargs):
        """Callbacks during training the models.

        Keyword Args:
            See `keras.callbacks.ModelCheckpoint`.

        Raises:
            NotImplementedError: `keras.save_model` hasn't been
                implemented for model subclassing.

        Returns:
            list: List of callbacks.
        """

        # Saving model isn't implemented yet for subclassed models in Keras.
        # raise NotImplementedError(
        #     "`keras.save_model` hasn't been implemented for model subclassing."
        # )

        # Saved model filepath.
        filepath = f'{ self._model_dir}/model-{"epoch:03d"}.h5'

        # Defaults to save best model.
        kwargs.setdefault('save_best_only', True)

        # Model checkpoint callback.
        checkpoint = keras.callbacks.ModelCheckpoint(filepath, **kwargs)

        return [checkpoint]

    def to_estimator(self):
        """Convert this model to `tf.estimator`.

        Returns:
            tf.estimator: Estimator model of current keras model.
        """
        # Current Keras model.
        keras_model = self._model

        # Convert keras model to `tf.estimator`.
        self._model = keras.estimator.model_to_estimator(keras_model=keras_model,
                                                         model_dir='saved/models/estimator')

        # Current object is now a tf.estimator object.
        self.is_estimator = True

        # Return estimator model.
        return self._model

    def save_model(self, weights_only=False):
        """Save model's parameters or weights only to an h5 file.

        Args:
            weights_only (bool, optional): Defaults to False. If set to true,
                only model's weights will be saved.
        """

        # Pretty prints.
        self._log(f'\n{"-" * 65}\nSaving model...')

        if weights_only:
            # Save model weights.
            self._model.save_weights(filepath=self._save_path, overwrite=True)
        else:
            # Save entire model.
            self._model.save(filepath=self._save_path, overwrite=True)

        # Pretty prints.
        self._log(f'Saved model weights to "{self._save_path}"!\n{"-" * 65}\n')

    def load_model(self):
        """Load a saved model.

        Raises:
            FileExistsError: Model already saved to `Network.save_path`.

        Returns:
            keras.Model: Saved model.
        """

        if tf.gfile.Exists(self._save_path):
            self._log(f'Loading model from {self._save_path}')
            self._model = keras.models.load_model(self._save_path)
        else:
            raise FileNotFoundError(f'{self._save_path} was not found.')

        return self._model

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
    def dist_func(x):
        """Difference function. Compute difference between 2 images.

        Args:
            x (tf.Tensor): Signifying two inputs.

        Returns:
            tf.Tensor: Absolute squared difference between two inputs.
        """

        return tf.square(tf.abs(x[0] - x[1]))

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
        """Network's background model.

        Returns:
            keras.Model: Underlaying model used by Network.
        """

        return self._model

    @property
    def save_path(self):
        """Path to saved model/model's weight.

        Returns:
            str: Path to an h5 file.
        """

        return self._save_path

    @property
    def model_dir(self):
        """Directory for saving and loading model.

        Returns:
            str: Path to an h5 file.
        """

        return self._model_dir
