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
    generate_more_image_no_middle()

def generate_more_image_no_middle():
    """produce more image by fading out"""
    step = 10
    training_set, validation_set, test_set = mnist_loader.load_percent_data(
        seed=0, percentage=0.01)
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
                transitional_images = [x1 + dx*k for k in range(1, step) if k<=3 or k>=7]
                expanded_images += transitional_images
                expanded_labels += [digit] * (step-4)


    all_images = np.array(expanded_images + list(training_set[0]))
    all_labels = np.array(expanded_labels + list(training_set[1]))

    permutaed_index = np.random.permutation(len(all_images))
    all_images = all_images[permutaed_index]
    all_labels = all_labels[permutaed_index]

    new_training_set = (all_images, all_labels)
    print all_labels[:10]
    plot_images_together(all_images[:10])

    fp=gzip.open('data/mnist_1_percent_expanded_10_step_no_mid.pkl.gz','wb')
    pickle.dump((new_training_set, validation_set, test_set), fp)
    fp.close()


def generate_more_image():
    """produce more image by fading out"""
    step = 5
    training_set, validation_set, test_set = mnist_loader.load_percent_data(
        seed=0, percentage=0.01)
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
                transitional_images = [x1 + dx*k for k in range(1, step)]
                expanded_images += transitional_images
                expanded_labels += [digit] * (step-1)


    all_images = np.array(expanded_images + list(training_set[0]))
    all_labels = np.array(expanded_labels + list(training_set[1]))

    permutaed_index = np.random.permutation(len(all_images))
    all_images = all_images[permutaed_index]
    all_labels = all_labels[permutaed_index]

    new_training_set = (all_images, all_labels)
    print all_labels[:10]
    plot_images_together(all_images[:10])

    fp=gzip.open('data/mnist_1_percent_expanded_5_step.pkl.gz','wb')
    pickle.dump((new_training_set, validation_set, test_set), fp)
    fp.close()

def plot_fade_out_transition():
    training_set, validation_set, test_set = mnist_loader.load_percent_data(
        percentage=0.01)
    fig = plt.figure()
    step = 5
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
def plot_images_together(images):
    """ Plot a single image containing all six MNIST images, one after
    the other.  Note that we crop the sides of the images so that they
    appear reasonably close together."""
    fig = plt.figure()
    images_2d = [np.reshape(image, (-1, 28)) for image in images]


    images = [image[:, 3:25] for image in images_2d]
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
