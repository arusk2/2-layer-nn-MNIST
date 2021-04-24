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

        print("training 1")
        self.forwardProp(inputs[0])
        self.checkCorrect(inputs[0],labels[0])

    def forwardProp(self, input):
        # get activation of hidden layer, excluding bias node.
        self.hidden_units[:self.nhidden] = np.dot(input, self.weights1)
        self.output = np.dot(self.hidden_units, self.weights2)

    def checkCorrect(self, input, label):
        # If correct, return, move onto next input. If wrong, do backprop
        prediction = np.argmax(self.output)
        # Set max prediction to .9
        self.output[prediction] = .9
        # convert outputs and labels to sigmoid activations
        self.output = np.where(self.output == .9, .9, .1)
        label = np.where(label == 1., .9, .1)
        if prediction != np.argmax(label):
            self.backProp(input, label)
        # Return if correct
        return

    def backProp(self, input, label):
        # We got the prediction wrong, need to backprop
        # First, set output for sigmoid fn ( .9 already set for max prediction in checkCorrect() )
        self.output = np.where(self.output == .9, .9, .1)

        # Update Weights for Hidden Units to Output:
        #   Delta(Wkj) = eta * sigk * Hj + momentum * (deltaT-1(Wkj))
        # Calculate error term sigk for output units, will add Hj as weights are updating.
        sigmak = self.output(1 - self.output)(label-self.output) # should be 1x10
        # Np.outer gives 10xn+1 matrix of Each Sigma multiplied by each Hj, resulting in a matrix
        # of sigmak * hj for all combinations of sigmak and hj such that row 1 is W1*H1 ... W1Hn
        # self.delta_tm1_wkj is a n+1x10 matrix, so I had to transpose the np.outer one to do elementwise add
        delta_Wkj = self.eta * np.transpose(np.outer(sigmak, self.hidden_units)) + (self.momentum*self.delta_tm1_Wkj)
        # ^ this should currently be n+1x10
        self.weights2 = self.weights2 - delta_Wkj
        #  ^ should be n+1x10
        self.delta_tm1_Wkj = delta_Wkj
        # ^ should be n+1x10

        # Now Updating weights from input to hidden (Wij)
        # when updating input to hidden weights, make sure to stop at hidden[:nhidden] to avoid bias
        # self.hidden = 1x20 vector; self.weights2 = n+1x10, np.transpose(sigmak) = 10x1
        # sigmaj should be 1x20 vector, each sum of all weights to hidden unit j
        sigmaj = self.hidden_units (1-self.hidden_units) * np.dot(self.weights2, np.transpose(sigmak))
        #                               V sigmaj now 20x1   V 1x785
        delta_Wij = self.eta * np.transpose(np.dot(np.transpose(sigmaj)), input) + (self.momentum * self.delta_tm1_Wij)
        #                       ^ Makes this array 785xn+1, just like delta_tm1_Wij
        self.weights1 = self.weights1 - delta_Wij
        return