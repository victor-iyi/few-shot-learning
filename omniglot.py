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

# Number of runs & example in each training, test folder.
n_runs = n_examples = 20


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


def visualize(test_dir: str, train_dir: str=None, **kwargs):
    """Visualize image groups with a matching 2 matching image.

    Args:
      test_dir (str): Test data directory.
      train_dir (str, optional): Defaults to None. Training data directory.

    Keyword Args:
      test_img (np.ndarray, optional): Defaults to None.
      train_img (np.ndarray, optional): Defaults to None.
      smooth (bool, optional): Defaults to True.
      cmap (str, optional): Defaults to gray.

      See `plt.imshow`
    """

    # Keyword arguments.
    test_img = kwargs.setdefault('test_img', None)
    train_img = kwargs.setdefault('train_img', None)

    smooth = kwargs.setdefault('smooth', True)
    cmap = kwargs.setdefault('cmap', 'gray')

    # Remvoe arguments irrelevant for matplotlib.
    kwargs.pop("smooth")
    kwargs.pop("test_img")
    kwargs.pop("train_img")

    # Update keyword arguments for matplotlib.
    kwargs.update({
        # CMap.
        "cmap": cmap,
        # Adjust smoothing interpolation.
        "interpolation": 'spline16' if smooth else 'nearest',
    })

    # Load images.
    test = np.array([load_image(os.path.join(test_dir, f))
                     for f in os.listdir(test_dir)])

    if test_dir is not None:
        train = np.array([load_image(os.path.join(train_dir, f))
                          for f in os.listdir(train_dir)])

    assert len(train) == len(test) == 20, 'train & test must be same length.'

    # Entire figure.
    gs = gridspec.GridSpec(1, 2)

    # Containing train_img and test_img.
    gs_label = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0])

    # Containing matching images from train_img & test_img.
    test_img_ax = plt.subplot(gs_label[0, 0])
    train_img_ax = plt.subplot(gs_label[1, 0])

    # Matching train & test images.
    test_img_ax.imshow(test_img, **kwargs)
    train_img_ax.imshow(train_img, **kwargs)

    # Remove ticks.
    test_img_ax.set_xticks([])
    test_img_ax.set_yticks([])
    train_img_ax.set_xticks([])
    train_img_ax.set_yticks([])

    # Containing all images from train_dir and test_dir
    gs_imgs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1])

    n_rows, n_cols = 5, 4
    # Containing all images from test_dir.
    gs_imgs_test = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols,
                                                    subplot_spec=gs_imgs[0])
    # Containing all images from train_dir.
    gs_imgs_train = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols,
                                                     subplot_spec=gs_imgs[1])

    idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            ax_test = plt.subplot(gs_imgs_test[row, col])
            ax_test.imshow(test[idx], **kwargs)

            ax_test.set_xticks([])
            ax_test.set_yticks([])

            ax_train = plt.subplot(gs_imgs_train[row, col])
            ax_train.imshow(train[idx], **kwargs)

            ax_train.set_xticks([])
            ax_train.set_yticks([])

            idx += 1

    # # Image dimensions.
    # img_shape = train.shape
    # img_batch, img_size = img_shape[0], img_shape[1]
    # img_channel = img_shape[-3] if len(img_shape) > 3 else 1

    # # Create figure with sub-plots.
    # fig, axes = plt.subplots(5, 4)

    # # Adjust vertical spacing if we need to print ensemble and best-net.
    # wspace, hspace = 0.2, 0.4
    # fig.subplots_adjust(hspace=hspace, wspace=wspace)

    # for i, ax in enumerate(axes.flat):
    #     # cmap type.
    #     cmap = 'gray' if img_channel == 1 else None
    #     # Interpolation type.
    #     smooth = 'spline16' if smooth else 'nearest'

    #     # Reshape image based on channel.
    #     if img_channel == 1:
    #         img = train[i].reshape(img_size, img_size)
    #     else:
    #         img = np.transpose(train[i], (1, 2, 0))

    #     # Plot image.
    #     ax.imshow(img, interpolation=smooth, cmap=cmap)

    #     # Remove ticks from the plot.
    #     ax.set_xticks([])
    #     ax.set_yticks([])

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
    run_dir = os.path.join(data_dir, 'run01')
    test_dir = os.path.join(run_dir, 'test')
    train_dir = os.path.join(run_dir, 'training')
    label_path = os.path.join(run_dir, 'class_labels.txt')

    # test_file = 'all_runs/run01/test/item01.png'
    # train_file = 'all_runs/run01/training/class_labels.png'
    # image = load_image(test_file)
    # imshow(image)
    # images = np.array([load_image(os.path.join(test_dir, f))
    #                    for f in os.listdir(test_dir)])
    with open(label_path, mode='r') as f:
        class_labels = f.readlines()

    test_img_path, train_img_path = class_labels[0].split()

    test_img_path = os.path.join(data_dir, test_img_path)
    train_img_path = os.path.join(data_dir, train_img_path)

    test_img = load_image(test_img_path)
    train_img = load_image(train_img_path)

    visualize(train_dir=train_dir, test_dir=test_dir,
              test_img=test_img, train_img=train_img)
