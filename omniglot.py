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
import pickle
import zipfile
import tarfile

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PIL import Image
from sklearn.utils import shuffle

# Base data & save directory.
base_dir = 'datasets/'
save_dir = 'saved/'

# Data sub directories.
data_dir = os.path.join(base_dir, 'extracted')
compressed_dir = os.path.join(base_dir, 'compressed')

# Number of runs & example in each training, test folder.
n_runs = n_examples = 20


class Visualize(object):

    @staticmethod
    def image(image: np.ndarray, title: str, **kwargs):
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
    def runs(directory: str, index: int=1, title: str='', **kwargs):
        # Keyword arguments.
        kwargs.setdefault('cmap', 'gray')
        smooth = kwargs.setdefault('smooth', True)

        # Update keyword arguments by matplotlib.
        kwargs.pop("smooth")
        kwargs.update({"interpolation": 'spline16' if smooth else 'nearest'})

        # Run files and sub directories.
        test_dir = os.path.join(directory, 'test')
        train_dir = os.path.join(directory, 'training')
        label_path = os.path.join(directory, 'class_labels.txt')

        # Load images.
        test = np.array([Data.load_image(os.path.join(test_dir, f))
                         for f in os.listdir(test_dir) if f[0] is not '.'])

        train = np.array([Data.load_image(os.path.join(train_dir, f))
                          for f in os.listdir(train_dir) if f[0] is not '.'])

        assert len(train) == len(test) == n_examples, \
            f'{len(train)}, {len(test)} and {len(n_examples)} are not equal'

        # Class labels.
        with open(label_path, mode='r') as f:
            class_labels = f.readlines()

        # Get class label to focus on.
        test_path, train_path = class_labels[index].split()
        test_path = os.path.join(os.path.dirname(directory), test_path)
        train_path = os.path.join(os.path.dirname(directory), train_path)

        # Load emphasized images.
        test_img = Data.load_image(test_path)
        train_img = Data.load_image(train_path)

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

    @staticmethod
    def symbols(directory: str, **kwargs):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f'{directory} is not a valid directory!')

        # Extract keyword arguments.
        smooth = kwargs.setdefault('smooth', True)
        cmap = kwargs.setdefault('cmap', 'gray')

        # Arguments used by matplotlib.
        kwargs.pop('smooth')
        kwargs.update({'interpolation': 'spline16' if smooth else 'nearest'})

        # Get the symbol name.
        title = os.path.basename(directory).replace('_', ' ')

        # List of characters.
        chars = (os.path.join(directory, c)
                 for c in os.listdir(directory) if c[0] is not '.')

        # Pick one character at random for each classes.
        img_paths = (os.path.join(ch, os.listdir(ch)[np.random.choice(n_examples)])
                     for ch in chars)

        # Load images.
        images = [Data.load_image(p) for p in img_paths]

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


class Data(object):

    class Mode:
        """Data.Mode - Dataset pre-processing mode."""
        TRAIN = "TRAIN"
        TEST = "TEST"
        VAL = "VALIDATE"
        VALIDATE = VAL

    def __init__(self, **kwargs):
        pass

    def __repr__(self):
        return 'omniglot.Data()'

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def extract(path: str, extract_dir=data_dir):
        # Ensure the file exists.
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Could not find {path}')

        # Create extract directory if it doesn't exist.
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        if zipfile.is_zipfile(path):
            print(f'Extracing {path}...')
            # Extract zipped file.
            with zipfile.ZipFile(path, mode="r") as z:
                z.extractall(data_dir)
        elif path.endswith((".tar.gz", ".tgz")):
            print(f'Extracing {path}...')
            # Unpack tarball.
            with tarfile.open(path, mode="r:gz") as t:
                t.extractall(data_dir)
        else:
            # Unrecognized compressed file.
            raise ValueError(f'{path} must a zipped or tarball file')

        # Retrive extracted directory.
        extracted_dir = os.path.basename(path).split('.')[0]
        extracted_dir = os.path.join(data_dir, extracted_dir)

        # Display & return extracted directory.
        print(f'Sucessfully extracted to {extracted_dir}')
        return extracted_dir

    @staticmethod
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
            raise FileNotFoundError(f'{path} was not found!')
        except Exception as e:
            raise Exception(f'ERROR: {path}')

        # Convert Pillow object to NumPy array.
        image = np.array(image, dtype=dtype)

        if flatten:
            # Flatten image.
            image = image.ravel()

        return image

    @staticmethod
    def get_images(directory: str=None, paths: str=None,
                   dtype: np.dtype=np.float32):
        if directory is not None:
            images = [Data.load_image(p) for p in Data._listdir(directory)]
        elif paths is not None:
            images = [Data.load_image(p) for p in paths]
        else:
            raise ValueError('Either `directory` or `paths` must be provided!')

        images = np.array(images, dtype=dtype)
        return images

    @staticmethod
    def _listdir(root, tolist=False):
        if tolist:
            # List comprehension.
            return [os.path.join(root, f) for f in os.listdir(root)
                    if f[0] is not '.']

        # Generator expression.
        return (os.path.join(root, f) for f in os.listdir(root)
                if f[0] is not '.')

    @staticmethod
    def _filter_files(files: iter):
        ignored_list = ('', '.DS_Store')

        def ignore(x: str):
            return x not in ignored_list or not x.endswith(('.jpg', '.png'))

        return list(filter(ignore, files))


class Dataset(Data):

    def __init__(self, path: str=None, mode=None, **kwargs):

        # Use argument or `omniglot.data_dir`.
        path = path or data_dir
        self._mode = mode or Dataset.Mode.TRAIN

        cache_dir = os.path.join(save_dir, os.path.basename(path).split('.')[0],
                                 self._mode.lower())

        # Extract keyword arguments.
        self._cache = kwargs.get('cache', True)
        self._cache_dir = kwargs.get('cache_dir', cache_dir)
        self._verbose = kwargs.get('verbose', 1)

        if self._cache:
            if not os.path.isdir(self._cache_dir):
                self._log(f'Creating {self._cache_dir}...\n')
                os.makedirs(self._cache_dir)

        if os.path.isdir(path):
            # Pre-process directory into pickle.
            self._data_dir = path
            self._images, self._targets = self.load(self._data_dir)
        elif path.endswith((".zip", ".gz", ".tar.gz")):
            self._data_dir = Dataset.extract(path)
            # Pre-process directory to be pickled.
            self._images, self._targets = self.load(self._data_dir)
        elif path.endswith((".pkl", ".pickle")):
            pass
        else:
            raise FileNotFoundError(f'{path} was not found.')

        # self.n_examples, self.n_classes, self._width, self._height = self._images.shape
        self.n_classes, self.n_examples, self._width, self._height = self._images.shape
        self._channel = 1

    def __repr__(self):
        return (f"Dataset(mode='{self._mode}', cache={self._cache}, "
                f"cache_dir='{self._cache_dir}')")

    def __getitem__(self, idx: int):
        pass

    def __len__(self):
        return self.len(self._images)

    @classmethod
    def from_cache(cls, path: str):
        if not os.path.isfile(path):
            FileNotFoundError(f"{path} not found")

        if not path.endswith(('.pkl', '.pickle')):
            raise ValueError("{path} is not a pickle file")

        with open(path, mode='rb') as f:
            inst = pickle.load(f)

        return inst

    @classmethod
    def from_xy(cls, X: np.ndarray, y: np.ndarray):
        # Create an instance..
        inst = cls()

        # Set X & Y.
        inst._images, inst._images = x, y

        return inst

    def load(self, path: str, dtype: np.dtype=np.float32):
        X = self.read("images")
        y = self.read("targets")

        # Return cached images & labels.
        if X is not False and y is not False:
            self._log('Loading cached images & corresponding targets.')

            return X, y

        # Process "images" & "targets".
        self._log('Loading images & targets')

        # Images, labels & class indices.
        X, y, class_idx = [], [], 0

        # Table header.
        print(f'ID  Alphabet {"Status":>42}')

        # Keep track of anything Exception.
        status = True
        for i, (root, folder, files) in enumerate(os.walk(path)):
            # Filter files that aren't images.
            files = self._filter_files(files)

            if len(files) > 1:
                try:
                    # Get ||image file names||.
                    img_paths = [os.path.join(root, f) for f in files]

                    # Images & class index.
                    X.append(self.get_images(paths=img_paths))
                    y.append(class_idx)

                    # Everything went okay.
                    status = True
                except Exception as e:
                    # Something went wrong!
                    print(f'ERROR: {e}')
                    status = False
            else:
                # Continue if this is not an Alphabet folder.
                if not folder[0].startswith("character"):
                    continue

                # Get Alphabet's name.
                if self._verbose:
                    name = os.path.basename(root).replace('_', ' ')
                    print(
                        f'{class_idx:02d}. {name:<45} {"DONE" if status else "ERROR"}')

                # Increment class index.
                class_idx += 1

        # Images & targets.
        X, y = np.stack(X), np.vstack(y)

        if self._cache:
            # Save images & targets.
            self.save(X, 'images')
            self.save(y, 'targets')

        self._log(f'\nImages = {X.shape}\tTargets = {y.shape}\n')

        return X, y

    def next_batch(self, batch_size: int=128):
        """Batch generator. Gets the next image pairs and corresponding target.

            Args:
                batch_size (int, optional): Defaults to 128. Mini batch size.

            Yields:
                tuple (pairs, target) -- Image pairs & target (0 or 1)
                    target=1 if pairs are the same letter & 0 otherwise.
        """

        while True:
            pairs, target = self.get_batch(batch_size=batch_size)
            yield pairs, target

    def get_batch(self, batch_size: int=128):
        """Get a randomly sampled mini-batch of image pairs and corresponding targets.

        Args:
            batch_size (int, optional): Defaults to 128. Mini-batch size.

        Returns:
            tuple: image pairs and respective targets.
        """

        # Shorten usage of random number generator.
        rand = np.random.randint

        # Shorten usage of image dimension.
        img_dim = self._width, self._height, self._channel

        # Half `batch_size`.
        half_batch = batch_size // 2

        # Randomly sample several classes (alphabet) to use in the batch.
        categories = np.random.choice(self.n_classes, size=(batch_size,))

        # Initialize 2 empty arrays for the input image batch.
        # pairs = np.zeros(shape=(2, batch_size, *img_dim))
        first = np.zeros(shape=(batch_size, *img_dim))
        second = np.zeros(shape=(batch_size. *img_dim))

        # Initialize vector for the targets, and make one half
        # of it '1's, so 2nd half of batch has same class.
        targets = np.zeros(shape=(batch_size,))
        targets[half_batch:] = 1

        for i in range(batch_size):
            # Pick the i'th random class (alphabet).
            cat1 = categories[i]

            # For 1st image pair:
            # Sample a character ID from characters in this category.
            idx1 = rand(low=0, high=self.n_examples)
            first[i, :, :, :] = self._images[cat1, idx1].reshape(img_dim)

            # For 2nd image pair:
            idx2 = rand(low=0, high=self.n_examples)

            # Pick images of same class for 1st half, different for 2nd half.
            if i >= half_batch:
                cat2 = cat1
            else:
                # Add a random number to the category modulo n classes to ensure
                # 2nd image has different category.
                cat2 = (cat1 + rand(1, self.n_classes)) % self.n_classes
            second[i, :, :, :] = self._images[cat2, idx2].reshape(img_dim)

        pairs = [first, second]

        return pairs, targets

    def one_shot_task(self, N: int):
        """Create image pair for N-way one shot learning task.

        Args:
            N (int): Number of pairs to generate.
        """

        indicies = np.random.randint(0, self.n_examples, size=(N,))

        # Pick random letters.
        categories = np.random.randint(self.n_classes, size=(N,))

        true_cat = categories[0]
        ex1, ex2 = np.random.choice(self.n_examples, size=(2,))

        # 1st image in pair.
        first = np.array([self._images[true_cat, ex1, :, :, ]] * N)
        first = first.reshape(N, self._width, self._height, self._channel)

        # 2nd image in pair (support set).
        second = self._images[categories, indicies, :, :]
        second[0, :, :] = self._images[true_cat, ex2]
        second = second.reshape(N, self._width, self._height, self._channel)

        # Target
        targets = np.zeros(shape=(N,))
        targets[0] = 1

        # Shuffle data.
        targets, first, second = shuffle(targets, first, second)

        pairs = [first, second]

        return pairs, targets

    def save(self, obj: any, name: str):
        """Save object for easy retrival.

        Args:
            obj (any): Object to be cached.
            name (str): base name of object.

        Raises:
            UserWarning: `Dataset.cache` is set to False.
        """

        # Warn user about caching when cache is set ot False.
        if not self._cache:
            raise UserWarning("`Dataset.cache` is set to False.")

        path = os.path.join(self._cache_dir, name)

        if isinstance(obj, np.ndarray):
            path = f'{path}.npy'
            np.save(path, obj)
        else:
            path = f'{path}.pkl'
            with open(path, mode="wb") as f:
                pickle.dump(obj, file)

        self._log(f'Cached "{name}" to "{path}"')

    def read(self, name: str):
        npy_path, pkl_path = f'{name}.npy', f'{name}.pkl'

        npy_path = os.path.join(self._cache_dir, npy_path)
        pkl_path = os.path.join(self._cache_dir, pkl_path)

        if os.path.isfile(npy_path):
            # Load numpy object.
            obj = np.load(npy_path)
        elif os.path.isfile(pkl_path):
            # Load pickled.
            with open(pkl_path, mode="rb") as f:
                obj = pickle.load(f)
        else:
            # File not found.
            obj = False

        return obj

    def to_cache(self):
        pass

    def _log(self, *args, **kwargs):
        """Custom logger method for debug purposes."""

        if self._verbose:
            print(*args, **kwargs)

    @property
    def shape(self):
        _shape = (self.n_classes, self._width, self._height, self._channel)
        return _shape

    @property
    def images(self):
        return self._images

    @property
    def targets(self):
        return self._targets


if __name__ == '__main__':
    # test = Dataset(path=os.path.join(data_dir, 'images_evaluation'))

    train = Dataset(path=os.path.join(data_dir, 'images_background'))
    pairs, target = train.get_batch(batch_size=6)
    print(pairs[0].shape, pairs[1].shape, target.shape)
