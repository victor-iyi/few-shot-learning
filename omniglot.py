"""Omniglot dataset helper file.

  @description
    For visualizing, pre-processing and loading the Omniglot dataset.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola/

   @project
     File: omniglot.py
     Created on 18 May, 2018 @ 5:26 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


# Data directory.
data_dir = 'all_runs/'


def load_image(path: str, dtype: np.dtype=np.float32,
               size: tuple=None, flatten: bool=False,
               grayscale: bool=False):
    """Load image pas a numpy array.

    Args:
      path (str): Path to image to be loaded.
      dtype (np.dtype, optional): Defaults to np.float32. NumPy array data type.
      size (tuple, None): Defaults to None. Image resize dimension.
      flatten (bool, optional): Defaults to False. Maybe flatten image.
      grayscale (bool, optional): Defaults to False. Convert image to grayscale or not.

    Raises:
      ImportError: Please make sure you have Pillow installed.
          Run `pip3 install Pillow` to install Pillow.

      FileNotFoundError: `path` was not found!
          Double check file path to make sure it exits, or
          use guard checks like `os.path.isfile(path)`

    Returns:
      array-like: Image as a numpy array with dimension 2D or 3D array
          *(-if image is colored or grayscale image)*.
          If `flatten` is True. Returns a 1D-array.
    """
    try:
        # Open image as a Pillow object.
        image = Image.open(path)

        # Resize image.
        if size is not None:
            image = image.resize(size)

        # Convert image to grayscale.
        if grayscale:
            image = image.convert('L')

    except ImportError:
        raise ImportError('Please make sure you have Pillow installed.')
    except FileNotFoundError:
        raise FileNotFoundError('{} was not found!'.format(path))
    except Exception as e:
        raise Exception('ERROR: {}'.format(e))

    # Convert Pillow object to NumPy array.
    image = np.array(image, dtype=dtype)

    if flatten:
        # Flatten image.
        image = image.ravel()

    return image


def visualize(image: np.ndarray, title: str=None, **kwargs):
    # Extract default keyword arguments.
    cmap = kwargs.get("cmap") or "gray"

    # Update keyword arguments.
    kwargs.update({"cmap": cmap})

    # Add image to plot.
    plt.imshow(image, **kwargs)

    if title:
        plt.title(title)
    plt.show()


if __name__ == '__main__':
    test_file = 'all_runs/run01/test/item01.png'
    train_file = 'all_runs/run01/training/class_labels.png'
    image = load_image(test_file)
    # print(image.shape)
    visualize(image)
