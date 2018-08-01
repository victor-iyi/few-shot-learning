"""Omniglot visualization helper file.

   @description
     For visualizing, pre-processing and loading the Omniglot dataset.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola/

   @project
     Package: omniglot
     File: visualize.py
     Created on 28 July, 2018 @ 11:37 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
# Built-in libraries.
import os

# Third party libraries.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Custom dataset helper file.
from omniglot.data import Dataset, n_examples


class Visualize(object):
    """Heler class for visualizing images in omniglot dataset."""

    @staticmethod
    def image(image: np.ndarray=None, filename: str=None, title: str='Omniglot Image', **kwargs):
        """Visualize a single image.

        Args:
            image (np.ndarray): Image as a numpy array
            title (str): Title of the plot.

        Keyword Args:
          cmap (plt.cmap, optional): Defaults to gray.
          smooth (bool, optional): Defaults to True. Whether to apply smoothening interpolation to image.

          See `matplotlib.pyplot.imshow`.

        """

        if image is None and filename is not None:
            # Read image from filename.
            image = Dataset.load_image(filename)

        # Make sure `image` is a np.ndarray.
        assert issubclass(image, np.ndarray), 'Image should be a numpy array'

        # Extract default keyword arguments.
        kwargs.setdefault('cmap', 'gray')
        smooth = kwargs.setdefault('smooth', False)

        # Interpolation type.
        smooth = 'spline16' if smooth else 'nearest'

        # Update keyword arguments.
        kwargs.pop('smooth')
        kwargs.update({"interpolation": smooth})

        # Add image to plot.
        plt.imshow(image, **kwargs)

        # Image title.
        plt.title(title)

        # Remove ticks from the plot.
        plt.xticks([])
        plt.yticks([])

        # Maybe show image.
        plt.show()

    @staticmethod
    def runs(directory: str, index: int = 1, title: str = 'One-shot Task', **kwargs):
        """Plot a single run in the omniglot "all_runs" data file.

        Args:
            directory (str): Directory to a single run.
            index (int, optional): Defaults to 1. Index to emphasized letter.
            title (str, optional): Defaults to ''. Title of the plot.

        Keyword Args:
            cmap (pyplot.cmap, optional): Defaults to 'gray'. See `matplotlib.pyplot.cmap`.
            smooth (bool, optional): Defaults to True. Smooth images.

            See `matplotlib.pyplot.imshow` for more options.
        """

        # Keyword arguments.
        kwargs.setdefault('cmap', 'gray')
        smooth = kwargs.setdefault('smooth', True)

        # Update keyword arguments by matplotlib.
        kwargs.pop("smooth")
        kwargs.update({
            # Change interpolation to smooth images.
            "interpolation": 'spline16' if smooth else 'nearest'
        })

        # Run files and sub directories.
        test_dir = os.path.join(directory, 'test')
        train_dir = os.path.join(directory, 'training')
        label_path = os.path.join(directory, 'class_labels.txt')

        # Load images.

        test = np.array([Dataset.load_image(f)
                         for f in Dataset._listdir(test_dir)])

        train = np.array([Dataset.load_image(f)
                          for f in Dataset._listdir(train_dir)])

        assert len(train) == len(test) == n_examples, \
            f'{len(train)}, {len(test)} and {len(n_examples)} are not equal.'

        # Class labels.
        with open(label_path, mode='r') as f:
            class_labels = f.readlines()

        # Get class label to focus on.
        test_path, train_path = class_labels[index].split()
        test_path = os.path.join(os.path.dirname(directory), test_path)
        train_path = os.path.join(os.path.dirname(directory), train_path)

        # Load emphasized images.
        test_img = Dataset.load_image(test_path)
        train_img = Dataset.load_image(train_path)

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

        # Containing all images from train_dir.
        gs_imgs_train = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols,
                                                         subplot_spec=gs_imgs[1])

        idx = 0
        for row in range(n_rows):
            for col in range(n_cols):
                # Plot test image on appropriate subplot.
                ax_test = plt.subplot(gs_imgs_test[row, col])
                ax_test.imshow(test[idx], **kwargs)

                # Remove x & y ticks from test subplot.
                ax_test.set_xticks([])
                ax_test.set_yticks([])

                # Plot train images on appropriate subplot.
                ax_train = plt.subplot(gs_imgs_train[row, col])
                ax_train.imshow(train[idx], **kwargs)

                # Remove x & y ticks from train subplot.
                ax_train.set_xticks([])
                ax_train.set_yticks([])

                # Increase indexing for train & test images.
                idx += 1

        # Entire figure's title.
        plt.suptitle(title)

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()

    @staticmethod
    def symbols(directory: str, **kwargs):
        """Plot all letters in a given letter directory.

        Args:
            directory (str): Path to an Omniglot letter directory.

        Keyword Args:
            cmap (pyplot.cmap, optional): Defaults to 'gray'. See `matplotlib.pyplot.cmap`.
            smooth (bool, optional): Defaults to True. Smooth images.

            See `matplotlib.pyplot.imshow` for more options.

        Raises:
            FileNotFoundError: `directory` is not a valid directory.
        """

        if not os.path.isdir(directory):
            raise FileNotFoundError(f'{directory} is not a valid directory!')

        # Extract & set default keyword arguments.
        smooth = kwargs.setdefault('smooth', True)
        kwargs.setdefault('cmap', 'gray')

        # Arguments used by matplotlib.
        kwargs.pop('smooth')
        kwargs.update({'interpolation': 'spline16' if smooth else 'nearest'})

        # Get the symbol name.
        title = os.path.basename(directory).replace('_', ' ')

        # Pick one character at random for each classes.
        img_paths = (os.path.join(ch, os.listdir(ch)[np.random.choice(n_examples)])
                     for ch in Dataset._listdir(directory))

        # Load images.
        images = [Dataset.load_image(p) for p in img_paths]

        # Visualize images.

        # Create figure with sub-plots.
        fig, axes = plt.subplots(5, 4)

        # Adjust vertical spacing.
        fig.subplots_adjust(hspace=0.4, wspace=0.2)

        # Plot images.
        for i, ax in enumerate(axes.flat):
            # Plot image on current axis.
            ax.imshow(images[i], **kwargs)

            # Remove x & y ticks.
            ax.set_xticks([])
            ax.set_yticks([])

        # Set plot's title & show figure.
        plt.suptitle(title)
        plt.show()

    @staticmethod
    def one_shot_task():
        pass
