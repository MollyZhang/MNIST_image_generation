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
import datetime
import gzip
import cPickle as pickle

def main():
    generate_more_image()



def generate_more_image():
    """produce more image by fading out"""
    step = 10
    training_set, validation_set, test_set = mnist_loader.load_1_percent_data(0)
    expanded_images = []
    expanded_labels = []
    for digit in range(10):
        images = get_images_by_digit_flat(training_set, digit)
        for i in range(len(images)):
            for j in range(i+1, len(images)):
                x1 = images[i]
                x2 = images[j]
                x = x2 - x1
                dx = x/step
                transitional_images = [x1 + dx*k for k in range(step)]
                expanded_images += transitional_images
                expanded_labels += [digit] * step

    all_images = np.array(expanded_images + list(training_set[0]))
    all_labels = np.array(expanded_labels + list(training_set[1]))

    all_images = all_images[np.random.permutation(len(all_images))]
    all_labels = all_labels[np.random.permutation(len(all_labels))]

    new_training_set = (all_images, all_labels)

    fp=gzip.open('data/mnist_1_perecent_expanded.pkl.gz','wb')
    pickle.dump((new_training_set, validation_set, test_set), fp)
    fp.close()

def plot_fade_out_transition():
    training_set, validation_set, test_set = mnist_loader.load_1_percent_data()
    fig = plt.figure()
    step = 10
    for digit in range(10):
        images = get_images_by_digit(training_set, digit)
        now = datetime.datetime.now().microsecond
        np.random.seed(now)
        i = np.random.randint(0, len(images))
        j = np.random.randint(0, len(images))
        x1 = images[i]
        x2 = images[j]
        x = x2 - x1
        dx = x/step
        x_series = [x1]
        for i in range(step):
            x_series.append(x1 + dx*i)
        x_series.append(x2)
        image = np.concatenate(x_series, axis=1)
        ax = fig.add_subplot(10, 1, digit+1)
        ax.matshow(image, cmap = matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.show()


#### Plotting
def plot_images_together(images, digit):
    """ Plot a single image containing all six MNIST images, one after
    the other.  Note that we crop the sides of the images so that they
    appear reasonably close together."""




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

def get_images_by_digit_flat(training_set, digit):
    """ Return a list containing the images from the MNIST data
    set. Each image is represented as a 2-d numpy array."""
    images = training_set[0]
    labels = training_set[1]
    digit_images = []
    for i in range(len(labels)):
        if labels[i] == digit:
            digit_images.append(images[i])
    return digit_images


#### Main
if __name__ == "__main__":
    main()
