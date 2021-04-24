# This is the perceptron code,
import numpy as np

class Perceptron():
    def __init__(self, nhidden, eta, momentum):
        self.nhidden = nhidden
        self.eta = eta
        self.momentum = momentum
        self.hidden_units = np.zeroes(nhidden+1)
        #set bias
        self.hidden_units[-1] = 1
        self.output = np.zeroes(10)

    # Train on whole set.
    def train(self, inputs, labels):
        # Pick inputs 1 by 1
        # input to hidden units
        # not adding weight to hidden unit bias, only to array of hidden units
        self.weights1 = np.random.rand((len(inputs[0]),self.nhidden)) - .5
        # hidden units (plus bias) to output
        self.weights2 = np.random.rand((self.nhidden+1, 10)) - .5
        # setting these to 0 for backprop later delta at time t - 1 for weights
        self.delta_tm1_Wkj = np.zeros((self.nhidden+1, 10))
        self.delta_tm1_Wij = np.zeros((len(inputs[0], self.nhidden)))
        pass

    def forwardProp(self, input):
        # get activation of hidden layer, excluding bias node.
        self.hidden_units[:self.nhidden] = np.dot(input, self.weights1)
        self.output = np.dot(self.hidden_units, self.weights2)

    def checkCorrect(self, label):
        # If correct, return, move onto next input. If wrong, do backprop
        prediction = np.argmax(self.output)
        # Set max prediction to .9
        self.output[prediction] = .9
        # convert outputs and labels to sigmoid activations
        self.output = np.where(self.output == .9, .9, .1)
        label = np.where(label == 1., .9, .1)
        if prediction != np.argmax(label):
            # back prop
        # Return if correct
        return

    def backProp(self, label):
        # We got the prediction wrong, need to backprop
        # First, set output for sigmoid fn ( .9 already set for max prediction in checkCorrect() )
        self.output = np.where(self.output == .9, .9, .1)

        # Update Weights for Hidden Units to Output:
        #   Delta(Wkj) = eta * sigk * Hj + momentum * (deltaT-1(Wkj))
        # Calculate error term sigk for output units, will add Hj as weights are updating.
        sigmak = self.output(1 - self.output)(label-self.output) # should be 1x10
        # array of weight kj delta, where k is output array. This array is how all weights should change
        # for each row j in weight array. Just need to multiply by value of each hidden unit corresponding
        # to all the weights we're updating.
        delta_Wkj = self.eta * sigmak
        for j in range(len(self.weights2)): #updating row j with 1x10 of k Wkj
            delta_Wkj += self.hidden_units[j]  # eta * sigk * Hj
            delta_Wkj += (self.momentum * self.delta_tm1_Wkj[j])  # + (momentum * deltaWt-1(kj)
            self.weights2[j] = self.weights2[j] - delta_Wkj
            self.delta_tm1_Wkj[j] = delta_Wkj  # Updating for next time step
        # when updating input to hidden weights, make sure to stop at hidden[:nhidden] to avoid bias
        pass