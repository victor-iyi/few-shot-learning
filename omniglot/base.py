"""Base class for building models for One-shot learning.

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

import tensorflow as tf
from tensorflow import keras
from abc import ABCMeta, abstractmethod

# Omniglot dataset helper class.
from omniglot import Dataset


# Classes in this file.
__all__ = ['BaseNetwork', 'Loss']


class Loss(object):
    """Implementation of popular loss function for One-shot learning tasks."""

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


class BaseNetwork(object):
    """Base class for building models for One-shot learning.

    Methods:
        @abstractmethod
        def call(self, inputs, **kwargs):
            Calls the model on new inputs. In this case `call` just reapplies
                all ops in the graph to the new inputs (e.g. build a new
                computational graph from the provided inputs).
            Args:
                inputs: A tensor or list of tensors.
            Keyword Args:
                training: Boolean or boolean scalar tensor, indicating whether
                    to run the `Network` in training mode or inference mode.
                mask: A mask or list of masks. A mask can be
                    either a tensor or None (no mask).
            Returns:
                A tensor if there is a single output, or
                a list of tensors if there are more than one outputs.

        @abstractmethod
        def build(self, **kwargs):
            Build the network architecture which returns a Keras model
            (instance of `keras.models.Model` class).
            Keyword Args:
                num_classes (int, optional): Defaults to 1. Number of output
                    classes.
            Raises:
                NotImplementedError: Sub-class must override `build` method.

        def train(self, train_data: Dataset, valid_data: Dataset=None,
                  batch_size: int=64, resume_training: bool=True, **kwargs):
            Train the initialized network/model on some dataset.
            Args:
              train_data (omniglot.Dataset): Training dataset. Must be an
                instance of the `omniglot.Dataset` class.
              valid_data (omniglot.Dataset, optional): Defaults to None.
                Validation dataset. Must be an instance of the
                `omniglot.Dataset` class.
              batch_size (int, optional): Defaults to 64. Training & validation
                mini-batch size.
              resume_training (bool, optional): Defaults to True. If set to
                `True`, training will be continued from previously saved h5 file
                (Network.save_path).
            Keyword Args:
              See `keras.models.Model.fit_generator` for more options.

        def callbacks(self, **kwargs):
            Callbacks during training the models.
            Keyword Args:
                See `keras.callbacks.ModelCheckpoint`.
            Raises:
                NotImplementedError: `keras.save_model` hasn't been implemented for
                    model subclassing.

        def to_estimator(self):
            Convert this model to `tf.estimator`.
            Returns:
                tf.estimator: Estimator model of current keras model.

        def save_model(self, weights_only=False):
            Save model's parameters or weights only to an h5 file.
            Args:
                weights_only(bool, optional): Defaults to False. If set to true,
                    only model's weights will be saved.

        def load_model(self):
            Load a saved model.
            Raises:
                FileExistsError: Model already saved to `Network.save_path`.
            Returns:
                keras.Model: Saved model.

        def _log(self, *args, **kwargs):
            Logging method helper based on verbosity.

        @staticmethod
        def dist_func(x):
            Difference function. Compute difference between 2 images.
            Args:
                x (tf.Tensor): Signifying two inputs.
            Returns:
                tf.Tensor: Absolute squared difference between two inputs.

    Attributes:
        model: Network's background model.
        save_path: Path to saved model/model's weight.
        model_dir: Directory for saving and loading model.

    Examples:
    ```python
    >>> network = EncoderNetwork(verbose=1)
    __________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================
    input_1 (InputLayer)            (None, 105, 105, 1)  0                                            
    __________________________________________________________________________________
    input_2 (InputLayer)            (None, 105, 105, 1)  0                                            
    __________________________________________________________________________________
    sequential (Sequential)         (None, 4096)         10636096    input_1[0][0]                    
                                                                     input_2[0][0]                    
    __________________________________________________________________________________
    lambda (Lambda)                 (None, 4096)         0           sequential[1][0]                 
                                                                     sequential[2][0]                 
    __________________________________________________________________________________
    dense_1 (Dense)                 (None, 1)            4097        lambda[0][0]                     
    ==================================================================================
    Total params: 10,640,193
    Trainable params: 10,640,193
    Non-trainable params: 0
    __________________________________________________________________________________

    >>> network.train(train_data=train_data, batch_size=128)
    Epoch 1/5
      3/128 [..............................] - ETA: 30:46 - loss: 0.6916 - acc: 0.4323
    Training interrupted by user!
    ----------------------------------------------------
    Saving model...
    Saved model weights to "saved/models/network.h5"!
    ----------------------------------------------------
    ```
    """
    # Abstract Base Class.
    __metaclass__ = ABCMeta

    # Loss functions for one-shot tasks.
    losses = Loss

    def __init__(self, num_classes=1, **kwargs):
        """`BaseNetwork.__init__(num_classes=1, **kwargs)`

        Args:
            num_classes (int, optional): Defaults to 1. Number of output classes
            in the last layer (prediction layer).

        Keyword Args:
            input_shape (tuple, optional): Defaults to (105, 105, 1). Input shape
                for a single image. Shape in the form: `(width, height, channel)`.

            lr (float, optional): Defaults to 1e-3. Optimizer's learning rate.

            verbose (int, optional): Defaults to 1. 0 or 1, if set to 0,
                progress or relevant  information will not be logged, otherwise
                relevant information will be logged.

            save_weights_only (bool, optional): Defaults to False. If set to `True`,
                only model's weights will be saved, otherwise the entire model is
                saved.

            optimizer (keras.optimizers.Optimizer, optional): Defaults to keras.optimizers.Adam.
                Network optimizer. See `keras.Model.compile`

            loss_func (Network.losses, optional): Defaults to BaseNetwork.losses.binary_crossentropy.

            model_dir (str, optional): Defaults to 'saved/models'. Directory
                where weights and model is saved.
        """
        # Extract Keyword arguments.
        lr = kwargs.get('lr', 1e-3)
        self._verbose = kwargs.get('verbose', 1)
        metrics = kwargs.get('metrics', ['accuracy'])
        self._input_shape = kwargs.get('input_shape', (105, 105, 1))
        self._save_weights_only = kwargs.get('save_weights_only', False)
        self.optimizer = kwargs.get('optimizer', keras.optimizers.Adam(lr=lr))

        self.loss_func = kwargs.get('loss_func', BaseNetwork.losses.binary_crossentropy)
        self._model_dir = kwargs.get('model_dir', 'saved/models/').rstrip('/')

        # Create model directory if it doesn't already exit.
        if not tf.gfile.IsDirectory(self._model_dir):
            tf.gfile.MakeDirs(self._model_dir)

        # Path to save model's weights.
        save_path = 'weights.h5' if self._save_weights_only else 'network.h5'
        self._save_path = f'{self._model_dir}/{save_path}'

        # Network built as a Keras Model.
        self.is_estimator = False

        # Instantiate a Sequential Model. [Can be overriden].
        self._model = self.build(**kwargs)

        # TODO: Get layerwise learning rates and momentum annealing scheme described in the paper.
        self._model.compile(loss=self.loss_func, optimizer=self.optimizer,
                            metrics=metrics)

        # Log summary if verbose is 'on'.
        self._log(callback=self._model.summary)

        # Parameter count.
        # n_params = self._model.count_params()
        # self._log(f'Network has {n_params:,} parameters.')

    def __repr__(self):
        return (f'Network(input_shape={self._input_shape}, '
                f'loss_func={self.loss_func}, optimizer={self.optimizer})')

    def __str__(self):
        return self.__repr__()

    def __call__(self, inputs, **kwargs):
        """Calls the model on new inputs.

        See `BaseNetwork.call`.
        """
        return self.call(inputs, **kwargs)

    @abstractmethod
    def call(self, **kwargs):
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
        raise NotImplementedError('Sub-class must override `call` method.')

    @abstractmethod
    def build(self, **kwargs):
        """Build the network architecture which returns a Keras model (instance
        of `keras.models.Model` class).

        Keyword Args:
            num_classes (int, optional): Defaults to 1. Number of output classes.

        Raises:
            NotImplementedError: Sub-class must override `build` method.

        Returns:
            keras.models.Model: Instance of Keras model.
        """

        raise NotImplementedError('Sub-class must override `build` method.')

    def train(self, train_data: Dataset, valid_data: Dataset=None,
              batch_size: int=64, resume_training: bool=True, **kwargs):
        """Train the initialized network/model on some dataset.

        Args:
            train_data (omniglot.Dataset): Training dataset. Must be an instance
                of the `omniglot.Dataset` class.
            valid_data (omniglot.Dataset, optional): Defaults to None.
                Validation dataset. Must be an instance of the
                `omniglot.Dataset` class.
            batch_size (int, optional): Defaults to 64. Training & validation
                mini-batch size.
            resume_training (bool, optional): Defaults to True. If set to `True`,
                training will be continued from previously saved h5 file
                `Network.save_path`.

        Keyword Args:
            See `keras.models.Model.fit_generator` for more options.

        Raises:
            See `keras.models.Model.fit_generator` for possible exceptions that
            might occur during training.
        """

        # Set default keyword arguments.
        kwargs.setdefault('epochs', 1)
        kwargs.setdefault('steps_per_epoch', 128)
        kwargs.setdefault('verbose', self._verbose)
        kwargs.setdefault('callbacks', self.callbacks())

        # Get batch generators.
        train_gen = train_data.next_batch(batch_size=batch_size)

        # Resume training.
        if resume_training and tf.gfile.Exists(self._save_path):
            self.load_model()

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

        # Save learned weights after completed training or KeyboardInterrupt.
        self.save_model()

    def callbacks(self, **kwargs):
        """Callbacks during training the models.

        Keyword Args:
            See `keras.callbacks.ModelCheckpoint`.

        Raises:
            NotImplementedError: `keras.save_model` hasn't been implemented for
                model subclassing.

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
                                                         model_dir='{self._model_dir}/estimator')

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