import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def prepare_data():

    data = input_data.read_data_sets("./MNIST_dataset", one_hot = True)

    print("Training set images shape: {}".format(data.train.images.shape))
    print("Training set labels shape: {}".format(data.train.labels.shape))
    print("Test set images shape: {}".format(data.test.images.shape))
    print("Test set labels shape: {}".format(data.test.labels.shape))

    img = np.reshape(data.train.images[50] , (28,28))
    plt.imshow(img)
    plt.show()
    print("Max pixel : {} \nMin pixel : {}".format(np.max(img) , np.min(img)))

    X_train = data.train.images.reshape(-1, 28 , 28 , 1)
    print(X_train.shape)

    X_test = data.test.images.reshape(-1, 28 , 28 , 1)
    print(X_test.shape)

    Y_train = data.train.labels

    Y_test = data.test.labels

    return X_train , Y_train , X_test , Y_test
