"""
mnist
~~~~~

Draws images based on the MNIST data."""

#### Libraries
# Standard library
import cPickle
import sys

# My library
import mnist_loader

# Third-party libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def main():
    training_set, validation_set, test_set = mnist_loader.load_1_percent_data()
    images = get_images_by_digit(training_set,1)

    x1 = images[0]
    x2 = images[1]
    x = x2 - x1
    plot_images_together(images[0:2] + [x])



#### Plotting
def plot_images_together(images):
    """ Plot a single image containing all six MNIST images, one after
    the other.  Note that we crop the sides of the images so that they
    appear reasonably close together."""
    fig = plt.figure()
    images = [image[:, 3:25] for image in images]
    image = np.concatenate(images, axis=1)
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()



#### Miscellanea
def load_data():
    """ Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data."""
    f = open('../data/mnist.pkl', 'rb')
    training_set, validation_set, test_set = cPickle.load(f)
    f.close()
    return (training_set, validation_set, test_set)

def get_images_by_digit(training_set, digit):
    """ Return a list containing the images from the MNIST data
    set. Each image is represented as a 2-d numpy array."""
    flattened_images = training_set[0]
    labels = training_set[1]
    images = []
    for i in range(len(labels)):
        if labels[i] == digit:
            images.append(np.reshape(flattened_images[i], (-1, 28)))
    return images

#### Main
if __name__ == "__main__":
    main()
