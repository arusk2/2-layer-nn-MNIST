# This is the perceptron code,
import numpy as np
from sklearn.utils import shuffle

class Perceptron():
    def __init__(self, epochs, nhidden, eta, momentum):
        self.nhidden = nhidden
        self.eta = eta
        self.momentum = momentum
        self.hidden_units = np.zeros((nhidden+1))
        #set bias
        self.hidden_units[-1] = 1
        self.output = np.zeros(10)
        self.epochs = epochs
        self.num_correct = 0
        # input to hidden units. Inputs have 785 features.
        # not adding weight to hidden unit bias
        self.weights1 = np.random.rand(785, self.nhidden) - .5
        self.delta_tm1_Wij = np.zeros((785, self.nhidden))
        # hidden units (plus bias) to output
        self.weights2 = np.random.rand(self.nhidden+1, 10) - .5
        # setting these to 0 for backprop later delta at time t - 1 for weights
        self.delta_tm1_Wkj = np.zeros((self.nhidden+1, 10))

    # this sigmoid function should allow me to calculate sigmoid activations for all hidden units
    def sigmoid(self, array):
        return 1 / (1 + np.exp(-array))

    # Train on whole set.
    def train(self, inputs, labels, test_inputs, test_labels):
        # Pick inputs 1 by 1
        train_acc = []
        test_acc = []
        for j in range(0, self.epochs):  # run training for 50 epochs
            # Shuffle inputs at each epoch to avoid overfitting
            inputs, labels = shuffle(inputs, labels, random_state=0)
            test_inputs, test_labels = shuffle(test_inputs, test_labels, random_state=0)
            # Resetting tm-1 deltas at each epoch
            # self.delta_tm1_Wij = np.zeros((785, self.nhidden))
            # self.delta_tm1_Wkj = np.zeros((self.nhidden + 1, 10))
            if j % 25 == 0:
                print("Epoch ", j)
            for i in range(0, len(inputs)):
                self.forwardProp(inputs[i])
                label = self.checkCorrect(labels[i])
                # label was copied in checkCorrect
                self.backProp(inputs[i], label)
            conf_matrix, tr_acc = self.test(inputs, labels)
            train_acc.append(tr_acc)
            conf_matrix, ts_acc = self.test(test_inputs, test_labels)
            test_acc.append(ts_acc)
        # return conf_matrix returns the final matrix of the test set.
        return conf_matrix, train_acc, test_acc

    def test(self, inputs, labels):
        # Uses code from Test Correct to get all the testing code in one function so I can easily track accuracy
        correct = 0
        conf_matrix = np.zeros((10,10))
        for i in range(0, len(inputs)):
            self.forwardProp(inputs[i])
            prediction = np.argmax(self.output)
            # Making a copy of the labels with sigmoid activation scheme applied (no 1s or 0s)
            if prediction == np.argmax(labels[i]):
                correct += 1
            conf_matrix[np.argmax(labels[i])][prediction] += 1
        return conf_matrix, correct/len(inputs)


    def forwardProp(self, input):
        # get activation of hidden layer, excluding bias node in hidden layer.
        self.hidden_units[:self.nhidden] = self.sigmoid(np.dot(input, self.weights1))
        self.output = self.sigmoid(np.dot(self.hidden_units, self.weights2))

    def checkCorrect(self, label):
        # Make prediction based on most activated output
        prediction = np.argmax(self.output)
        # adjust label for sigmoid function, making a copy instead of altering original
        sigmoid_label = np.where(label == 1, .9, .1)
        if prediction == np.argmax(sigmoid_label):
            self.num_correct += 1
        return sigmoid_label

    def backProp(self, input, label):
        # Update Weights for Hidden Units to Output:
        #   Delta(Wkj) = eta * sigk * Hj + momentum * (deltaT-1(Wkj))
        # Calculate error term sigk for output units, will add Hj as weights are updating.
        sigmak = self.output * (1 - self.output) * (label - self.output) # should be 1x10
        # transposing sigmak to get a 10x1, then dot producting it with hidden units(a 1x20
        # self.delta_tm1_wkj is a n+1x10 matrix, so I had to transpose the np.outer one to do elementwise add
        delta_Wkj = self.eta * np.transpose(np.dot(np.transpose([sigmak]), self.hidden_units.reshape(1, self.nhidden + 1))) + (self.momentum * self.delta_tm1_Wkj)
        # ^ this should currently be n+1x10
        self.weights2 = self.weights2 + delta_Wkj
        #  ^ should be n+1x10
        self.delta_tm1_Wkj = delta_Wkj
        # ^ should be n+1x10

        # Now Updating weights from input to hidden (Wij)
        # when updating input to hidden weights, make sure to stop at hidden[:nhidden] to avoid bias
        # self.hidden = 1x20 vector; self.weights2 = n+1x10, np.transpose(sigmak) = 10x1
        # sigmaj should be 1x20 vector, each sum of all weights to hidden unit j, not counting bias
        sigmaj = self.hidden_units[:self.nhidden] * (1-self.hidden_units[:self.nhidden]) * \
                 np.dot(self.weights2[:self.nhidden], np.transpose(sigmak))
        #                                                V sigmaj now nx1   V 1x785
        delta_Wij = self.eta * np.transpose(np.dot(np.transpose([sigmaj]), input.reshape(1, 785))) + (self.momentum * self.delta_tm1_Wij)
        #                       ^ Makes this array 785xn, just like delta_tm1_Wij

        self.weights1 = self.weights1 + delta_Wij
        self.delta_tm1_Wij = delta_Wij
        return