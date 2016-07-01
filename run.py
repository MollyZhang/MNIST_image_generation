import numpy as np
import pickle
import matplotlib.pyplot as plt
import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

def main():
    plot_result()

def train():
    training_data, validation_data, dummy = network3.load_data_shared(
        filename="data/mnist_1_percent_expanded_10_step.pkl.gz")

    mini_batch_size = 10
    net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                          filter_shape=(20, 1, 5, 5),
                          poolsize=(2, 2)),
            FullyConnectedLayer(n_in=20*12*12, n_out=100),
            SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    accuracies = net.SGD(training_data, 10000, mini_batch_size, 0.1, validation_data)

    with open("result/accuracy_1_percent_expanded_10_step.pkl", "w") as f:
        f.write(pickle.dumps(accuracies))
        f.close()

def plot_result():
    files = ["result/accuracy_1_percent.pkl",
             "result/accuracy_1_percent_expanded_5_step.pkl",
             "result/accuracy_1_percent_expanded_10_step.pkl"]
    accus = []
    for file in files:
        with open(file, "r") as f:
            accu = pickle.loads(f.read())
            accus.append(accu)

    x = range(100, 20100, 100)
    for accu in accus:
        print max(accu)

        plt.plot(x, accu[:200])
    plt.legend(["1% MNIST data",
                "1% MNIST data expanded by 5-step transition",
                "1% MNIST data expanded by 10-step transition"],
               loc = "best")
    plt.xlabel("number of mini batches")
    plt.ylabel("test accuray")
    plt.show()



if __name__ == '__main__':
    main()
    
    
