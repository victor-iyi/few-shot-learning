# Few Shot Learning

The process of learning good features for machine learning applications can be very computationally expensive and may prove difficult in cases where little data is available. A prototypical example of this is the *one-shot learning setting*, in which we must correctly make predictions given *only a single example* of each new class.

Here, I explored the power of *One-Shot Learning* with a popular model called "**Siamese Neural Network**".

## Table of Contents

- [Setup](#setup)
  - [Clone or Download](#clone-or-download)

- [Architecture](#architecture)
  - [Model](#model)
  - [Learning](#learning)

- [Credits](#credits)

- [Contribution](#contribution)

## Setup

### Clone or Download

Fire up your favorite command line utility *(e.g. Terminal, iTerm or Command Prompt)*, and type the following commands to clone the project.

```sh
$ git clone https://github.com/victor-iyiola/few-shot-learning.git
$ cd few-shot-learning && ls
LICENSE        README.md      datasets       images         omniglot       one-shot.ipynb          utils.py
```

or simply download this repository, and change directory to the downloaded project.

```sh
$ few-shot-learning$ ~ tree
something
```

## Architecture

### Model

Our standard model is a Siamese Convolutional Neural Network with $L$ layers each with $N_l$ units, where $h_{1, l}$ represents the hidden vector in layer $l$ for the first twin, and $h_{2,l}$ denotes the same for the second twin. We use exclusively **Rectified Linear (ReLU) Units** in the first $L-2$ layers and sigmoidal units in the remaining layers.

The model consists of a sequence of convolutional layers, each of which uses a single channel with filters of varying size and a fixed stride of 1. The number of convolutional filters is specified as a multiple of 16 to optimize performance. The network applies a ReLU activation function to the output feature maps, optionally followed by max-pooling with a filter size and stride of 2. Thus the $k^{th}$ filter map in each layer takes the following form:

$$ a^{(k)}_{1, m} = \textrm{max-pool}(max(0, W^{(k)}_{l-1} \star h_{1, (l-1)} + b_l), 2) $$

$$ a^{(k)}_{2, m} = \textrm{max-pool}(max(0, W^{(k)}_{l-1} \star h_{2, (l-1)} + b_l), 2) $$

where $W^{l-1, l}$ is the 3-dimensional tensor representing the feature maps for layer $l$ and we have taken $\star$ to be the valid convolutional operation corresponding to returning only those output units which were the result of complete overlap between each convolutional filter and the input feature maps.

![Best convolutional architecture selected for verification task. Siamese twin is not depicted, but joins immediately after the 4096 unit fully-connected layer where the L1 component-wise distance between vectors is computed.](images/Siamese%20Network.png)

### Learning

**Loss function.** Let $M$ represent the mini-batch size, where $i$ indexes the $i^{th}$ mini-batch. Now let $y(x^{(i)}_1, x^{(i)}_2)$ be a length-$M$ vector which contains the labels for the mini-batch, where we assume $y(x^{(i)}_1, x^{(i)}_2) = 1$ whenever $x_1$ and $x_2$ are from the same character class and $y(x^{(i)}_1, x^{(i)}_2) = 0$ otherwise. We impose a regularized cross-entropy objective on our binary classifier of the following form:

$$ \ell(x^{(i)}_1, x^{(i)}_2) = y(x^{(i)}_1, x^{(i)}_2)\log{p(x^{(i)}_1, x^{(i)}_2)} + (1 - y(x^{(i)}_1, x^{(i)}_2)) \log{(1 - p(x^{(i)}_1, x^{(i)}_2))} + \lambda^T\big|w\big|^2 $$

## Credits

- [Fellowship&period;ai](http://fellowship.ai)
- [Omniglot Dataset](https://github.com/brendenlake/omniglot)
- [Siamese Neural Networks for One-shot Image Recognition](http://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

## Contribution

This project is opened under the [MIT 2.0 License](https://github.com/victor-iyiola/few-shot-learning/blob/master/LICENSE) which allows very broad use for both academic and commercial purposes.

A few of the images used for demonstration purposes may be under copyright. These images are included under the "fair usage" laws.

You are very welcome to modify and use them in your own projects.
Please keep a link to the [original repository](https://github.com/victor-iyiola/few-shot-learning).

Pull requests and questions are welcomed. Please open an issue or [shoot me a mail](mailto:javafolabi@gmail.com) for any questions or further follow ups.
