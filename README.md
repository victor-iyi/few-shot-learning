# Few Shot Learning

The process of learning good features for machine learning applications can be very computationally expensive and may prove difficult in cases where little data is available. A prototypical example of this is the one-shot learning setting, in which we must correctly make predictions given only a single example of each new class.
I explored a method for learning Siamese Neural Networks which employ a unique structure to naturally rank similarity between inputs. Once a network has been tuned, we can then capitalize on powerful discriminative features to generalize the predictive power of the network not just to new data, but to entirely new classes from unknown distributions. Using a convolutional architecture, we are able to achieve strong results which exceed those of other deep learning models with near *state-of-the-art* performance on one-shot classification tasks.
Here, I explored the power of One-Shot Learning with a popular network called "**Siamese Neural Network**".

## Architecture

![Siamese Network](images/Siamese%20Network.png)

## Credits

- Omniglot Dataset
- Fellowship&period;ai
