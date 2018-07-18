"""Helper class for building Siamese model for One-shot learning.

   @description
     For visualizing, pre-processing and loading the Omniglot dataset.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola/

   @project
     File: model.py
     Created on 13 July, 2018 @ 9:10 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
import tensorflow as tf

from tensorflow import keras
import numpy as np


class SiameseNetwork(keras.Model):
    """Siamese Neural network for few shot learning."""

    def __init__(self, num_classes: int=1, **kwargs):
        super(SiameseNetwork, self).__init__(name='SiameseNetwork')

        # Positional Arguments.
        self.num_classes = num_classes

        # Keyword Arguments.
        self.batch_size = kwargs.get('batch_size', 16)
        self.in_shape = kwargs.get('input_shape', (105, 105, 1))

        # Input layer.
        self.input_layer = keras.layers.InputLayer(input_shape=self.in_shape,
                                                   batch_size=self.batch_size,
                                                   dtype=tf.float32, name='Images')

        # 1st layer (64@10x10)
        self.conv1 = keras.layers.Conv2D(filters=64, kernel_size=(10, 10),
                                         kernel_regularizer=keras.regularizers.l2,
                                         activation=keras.activations.relu)
        self.pool1 = keras.layers.MaxPool2D(pool_size=(2, 2))

        # 2nd layer (128@7x7)
        self.conv2 = keras.layers.Conv2D(filters=128, kernel_size=(7, 7),
                                         kernel_regularizer=keras.regularizers.l2,
                                         activation=tf.keras.activations.relu)
        self.pool2 = keras.layers.MaxPool2D(pool_size=(2, 2))

        # 3rd layer (128@4x4)
        self.conv3 = keras.layers.Conv2D(filters=128, kernel_size=(4, 4),
                                         kernel_regularizer=keras.regularizers.l2,
                                         activation=keras.activations.relu)
        self.pool3 = keras.layers.MaxPool2D(pool_size=(2, 2))

        # 4th layer (265@4x4)
        self.conv4 = keras.layers.Conv2D(filters=256, kernel_size=(4, 4),
                                         kernel_regularizer=keras.regularizers.l2,
                                         activation=keras.activations.relu)
        self.pool4 = keras.layers.MaxPool2D(pool_size=(2, 2))

        # 5th layer  (9216x4096)
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(units=4096,
                                        activation=keras.activations.sigmoid)

        # 6th - L1 layer -distance layer.
        self.distance = keras.layers.Lambda(self.dist_func)

        # Output layer (4096x1)
        self.prediction = keras.layers.Dense(units=self.num_classes,
                                             activation=keras.activations.sigmoid)

    def __repr__(self):
        return f'models.SiameseNetwork(num_classes={self.num_classes})'

    def call(self, inputs: tuple, **kwargs):
        """Calls the model on new inputs.

        In this case `call` just reapplies all ops in the graph to the new inputs
        (e.g. build a new computational graph from the provided inputs).

        Arguments:
            inputs: A tensor or list of tensors.
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
        distance = self.l1((first, second))

        # Prediction.
        pred = self.prediction(distance)

        return distance, pred

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
        loss = tf.maximum(y_ture - y_pred + alpha, 0)

        # Sum over all images.
        return tf.reduce_sum(loss, axis=1, name="Triplet_Loss")

    @staticmethod
    def constractive_loss(y_true, y_pred, alpha=0.2):
        """Constractive loss function.

        Binary cross entropy between the predictions and targets.
        There is also a L2 weight decay term in the loss to encourage
        the network to learn smaller/less noisy weights and possibly
        improve generalization:

        L(x1, x2, t) = t⋅log(p(x1 ∘ x2)) + (1−t)⋅log(1 − p(x1 ∘ x2)) + λ⋅||w||2

        Args:
            y_pred (tf.Tensor): Predicted distance between two inputs.
            y_true (tf.Tensor): Ground truth or target, t (where, t = [1 or 0]).

            alpha (float, optional): Defaults to 0.2. Slight margin
                added to prediction to avoid 0-learning.

        Returns:
            tf.Tensor: Constractive loss function.
        """

        loss = y_true * tf.log(y_pred) + (1 - y_true) * \
            tf.log(1 - y_pred) + alpha

        return tf.reduce_mean(loss, name="Constractive_Loss")

    @staticmethod
    def dist_func(x):
        """Difference function. Compute difference between 2 images.

        Args:
            x (tf.Tensor): Signifying two inputs.

        Returns:
            tf.Tensor: Absolute squared difference between two inputs.
        """

        return tf.abs(tf.squared_difference(x[0], x[1]))

    def __encoder(self, x):
        """Compute forward pass. Encoder part of the network.

        Args:
            x (tf.Tensor): Individual input to the SiameseNetwork.

        Returns:
            tf.Tensor: Encoded output.
        """
        # Input layer.
        x = self.input_layer(x)

        # Convolutional blocks.
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.pool4(self.conv4(x))

        # Flatten & fully connected layers.
        x = self.flatten(x)
        x = self.dense(x)

        return x

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)


if __name__ == '__main__':
    net = SiameseNetwork(num_classes=1)

    net.compile(optimizer=keras.optimizers.Adam(lr=1e-3),
                loss=SiameseNetwork.triplet_loss)
