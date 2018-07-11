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
import zipfile
import tarfile

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PIL import Image


# Data directory.
data_dir = 'datasets/'
extracted_dir = os.path.join(data_dir, 'extracted')
compressed_dir = os.path.join(data_dir, 'compressed')

# Number of runs & example in each training, test folder.
n_runs = n_examples = 20


def extract(path: str):
    # Ensure the file exists.
    if not os.path.isfile(path):
        raise FileNotFoundError('Could not find {}'.format(path))

    # Create extract directory if it doesn't exist.
    if not os.path.isdir(extracted_dir):
        os.makedirs(extracted_dir)

    if zipfile.is_zipfile(path):
        # Extract zipped file.
        with zipfile.ZipFile(path, mode="r") as z:
            z.extractall(extracted_dir)
    elif path.endswith((".tar.gz", ".tgz")):
        # Unpack tarball.
        with tarfile.open(path, mode="r:gz") as t:
            t.extractall(extracted_dir)
    else:
        # Unrecognized compressed file.
        raise Exception('{} must a zipped or tarball file'.format(path))

    print('Sucessfully extracted to {}'
          .format(os.path.join(extracted_dir, os.path.basename(path).split('.')[0])))


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


def visualize_runs(run_dir: str, index: int=0, title: str='', **kwargs):
    """Visualize image groups with a matching 2 matching image.

    Args:
      test_dir (str): Test data directory.

    Keyword Args:
      smooth (bool, optional): Defaults to True.
      cmap (str, optional): Defaults to gray.

      See `plt.imshow`
    """

    # Keyword arguments.
    test_img = kwargs.setdefault('test_img', None)
    train_img = kwargs.setdefault('train_img', None)

    smooth = kwargs.setdefault('smooth', True)
    cmap = kwargs.setdefault('cmap', 'gray')

    # Run files and sub directories.
    test_dir = os.path.join(run_dir, 'test')
    train_dir = os.path.join(run_dir, 'training')
    label_path = os.path.join(run_dir, 'class_labels.txt')

    # Load images.
    test = np.array([load_image(os.path.join(test_dir, f))
                     for f in os.listdir(test_dir)])

    train = np.array([load_image(os.path.join(train_dir, f))
                      for f in os.listdir(train_dir)])

    assert len(train) == len(test) == n_examples, \
        '{}, {} and {} are not equal'.format(len(train), len(test), n_examples)

    # Class labels.
    with open(label_path, mode='r') as f:
        class_labels = f.readlines()

    # Get class label to focus on.
    test_path, train_path = class_labels[index].split()
    test_path = os.path.join(os.path.dirname(run_dir), test_path)
    train_path = os.path.join(os.path.dirname(run_dir), train_path)

    # Load emphasized images.
    test_img = load_image(test_path)
    train_img = load_image(train_path)

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

    # Entire figure.
    gs = gridspec.GridSpec(1, 2)

    # Containing train_img and test_img.
    gs_label = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0])

    # Containing matching images from train_img & test_img.
    test_img_ax = plt.subplot(gs_label[0, 0])
    train_img_ax = plt.subplot(gs_label[1, 0])

    # Matching train & test images.
    test_img_ax.imshow(test_img, **kwargs)
    test_img_ax.set_xlabel('Test Handwriting')

    train_img_ax.imshow(train_img, **kwargs)
    train_img_ax.set_xlabel('Class target')

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
    # print(dir(gs_imgs_test.get_subplot_params()))
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
    # Entire figure's title.
    plt.suptitle(title)

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def imshow(image: np.ndarray, title: str='', **kwargs):
    # Extract default keyword arguments.
    cmap = kwargs.get('cmap') or 'gray'
    smooth = kwargs.setdefault('smooth', False)

    # Interpolation type.
    smooth = 'spline16' if smooth else 'nearest'

    kwargs.pop('smooth')

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
    plt.show()


if __name__ == '__main__':
    # Visualize single image.
    # test_file = 'all_runs/run01/test/item01.png'
    # image = load_image(test_file)
    # imshow(image, title=test_file, smooth=True)

    # Visualize single run.
    # run_dir = os.path.join(data_dir, 'all_runs/run01')
    # visualize_runs(run_dir, index=3, title='')
    all_runs = os.path.join(compressed_dir, 'all_runs.zip')
    extract(all_runs)
