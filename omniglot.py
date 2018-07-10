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
import matplotlib.gridspec as gridspec

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

# Helper function to plot images and labels.


def visualize(train_dir: str, test_dir: str=None, **kwargs):
    # Keyword arguments.
    smooth = kwargs.get('smooth', True)
    test_img = kwargs.get('test_img', None)
    train_img = kwargs.get('train_img', None)

    # Load images.
    train = np.array([load_image(os.path.join(train_dir, f))
                      for f in os.listdir(train_dir)])

    if test_dir is not None:
        test = np.array([load_image(os.path.join(test_dir, f))
                         for f in os.listdir(test_dir)])

    # Entire figure.
    gs = gridspec.GridSpec(1, 2)

    # Containing train_img and test_img.
    gs_label = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0])

    # Containing matching images from train_img & test_img.
    test_img_ax = plt.subplot(gs_label[0, 0])
    train_img_ax = plt.subplot(gs_label[1, 0])

    # Matching train & test images.
    test_img_ax.imshow(test_img)
    train_img_ax.imshow(train_img)

    # Containing all images from train_dir and test_dir
    gs_imgs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1])

    # Containing all images from test_dir.
    gs_imgs_test = gridspec.GridSpecFromSubplotSpec(5, 4,
                                                    subplot_spec=gs_imgs[0])
    # Containing all images from train_dir.
    gs_imgs_train = gridspec.GridSpecFromSubplotSpec(5, 4,
                                                     subplot_spec=gs_imgs[1])

    # Image dimensions.
    img_shape = train.shape
    img_batch, img_size = img_shape[0], img_shape[1]
    img_channel = img_shape[-3] if len(img_shape) > 3 else 1

    # Create figure with sub-plots.
    fig, axes = plt.subplots(5, 4)

    # Adjust vertical spacing if we need to print ensemble and best-net.
    wspace, hspace = 0.2, 0.4
    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    for i, ax in enumerate(axes.flat):
        # cmap type.
        cmap = 'gray' if img_channel == 1 else None
        # Interpolation type.
        smooth = 'spline16' if smooth else 'nearest'

        # Reshape image based on channel.
        if img_channel == 1:
            img = train[i].reshape(img_size, img_size)
        else:
            img = np.transpose(train[i], (1, 2, 0))

        # Plot image.
        ax.imshow(img, interpolation=smooth, cmap=cmap)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def imshow(image: np.ndarray, title: str='', **kwargs):
    # Extract default keyword arguments.
    cmap = kwargs.get('cmap') or 'gray'
    smooth = kwargs.get('smooth', False)
    show = kwargs.get('show', True)

    # Interpolation type.
    smooth = 'spline16' if smooth else 'nearest'

    # Update keyword arguments.
    kwargs.update({
        "cmap": cmap, "interpolation": smooth
    })

    # Add image to plot.
    plt.imshow(image, **kwargs)

    # Image title.
    plt.title(title)

    # Remove ticks from the plot.
    plt.xticks([])
    plt.yticks([])

    # Maybe show image.
    if show:
        plt.show()


if __name__ == '__main__':
    test_dir = os.path.join(data_dir, 'run01/test')

    test_file = 'all_runs/run01/test/item01.png'
    train_file = 'all_runs/run01/training/class_labels.png'
    # image = load_image(test_file)
    # imshow(image)
    images = np.array([load_image(os.path.join(test_dir, f))
                       for f in os.listdir(test_dir)])

    visualize(test_dir, title="Run 0")
