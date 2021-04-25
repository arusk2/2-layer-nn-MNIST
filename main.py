# experiment goes here. Another file has all the class stuff for perceptron
import keras
import numpy as np
from keras.datasets import mnist
import perceptron
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

# Experiment 1
nhidden = [20, 50, 100]
eta = .1
momentum = .9
epochs = 50
for n in nhidden:
    test = perceptron.Perceptron(epochs, n, eta, momentum)
    # train() shuffles inputs and so the order is not always the same
    conf_matrix, tr_acc, test_acc = test.train(xtrain, ytrain, xtest, ytest)
    tr_acc = np.array(tr_acc)
    tr_acc *= 100
    test_acc = np.array(test_acc)
    test_acc *= 100
    print(f"Final Testing Confusion Matrix for {n} Hidden Units.")
    print(f"Final Training Accuracy: {tr_acc[-1]}\tFinal Testing Accuracy: {test_acc[-1]}")
    print(conf_matrix.astype(int))
    plt.plot(tr_acc, label="Training Accuracy")
    plt.plot(test_acc, label="Testing Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Percent Correct')
    plt.title(f"Testing vs Training Accuracy with {n} Hidden Units")
    plt.legend()
    plt.show()
    plt.clf()

# Experiment 2
nhidden = 100
eta = .1
momentum = [0, .25, .5]
epochs = 50
for m in momentum:
    test = perceptron.Perceptron(epochs, nhidden, eta, m)
    # train() shuffles inputs and so the order is not always the same
    conf_matrix, tr_acc, test_acc = test.train(xtrain, ytrain, xtest, ytest)
    tr_acc = np.array(tr_acc)
    tr_acc *= 100
    test_acc = np.array(test_acc)
    test_acc *= 100
    print(f"Final Testing Confusion Matrix for {m} Momentum (100 Hidden Units)")
    print(f"Final Training Accuracy: {tr_acc[-1]}\tFinal Testing Accuracy: {test_acc[-1]}")
    print(conf_matrix.astype(int))
    plt.plot(tr_acc, label="Training Accuracy")
    plt.plot(test_acc, label="Testing Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Percent Correct')
    plt.title(f"Testing vs Training Accuracy with Momentum = {m} (100 Hidden Units)")
    plt.legend()
    plt.show()
    plt.clf()

# Experiment 3
nhidden = 100
eta = .1
momentum = .9
train = [.25, .5]
epochs = 50
for t in train:
    tr_c = int(t * len(xtrain))
    ts_c = int(t * len(xtest))
    test = perceptron.Perceptron(epochs, nhidden, eta, momentum)
    # train() shuffles inputs and so the order is not always the same
    conf_matrix, tr_acc, test_acc = test.train(xtrain[:tr_c], ytrain[:tr_c], xtest[:ts_c], ytest[:ts_c])
    tr_acc = np.array(tr_acc)
    tr_acc *= 100
    test_acc = np.array(test_acc)
    test_acc *= 100
    print(f"Final Testing Confusion Matrix for {tr_c} Training Examples.")
    print(f"Final Training Accuracy: {tr_acc[-1]}\tFinal Testing Accuracy: {test_acc[-1]}")
    print(conf_matrix.astype(int))

    plt.plot(tr_acc, label="Training Accuracy")
    plt.plot(test_acc, label="Testing Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Percent Correct')
    title = f"Testing vs Training Accuracy with {tr_c} Training Examples and {ts_c} Testing Examples (100 Hidden Units)"
    plt.title(title)
    plt.legend()
    plt.show()
    plt.clf()