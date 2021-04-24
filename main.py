# experiment goes here. Another file has all the class stuff for perceptron
import keras
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

#creating a perceptron network to classify MNIST data
(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
xtrain = xtrain.reshape(60000, 784)
xtest = xtest.reshape(10000, 784)
xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')
# normalize data
xtrain /= 255
xtest /= 255
# make input one hot
num_classes = 10
ytrain = keras.utils.to_categorical(ytrain, num_classes)
ytest = keras.utils.to_categorical(ytest, num_classes)
# add 1 for bias
ones = np.ones(60000)
xtrain = np.hstack((xtrain, np.atleast_2d(ones).T))
# ytrain =
ones = np.ones(10000)
xtest = np.hstack((xtest, np.atleast_2d(ones).T))
