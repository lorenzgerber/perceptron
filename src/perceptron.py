import numpy as np
from random import random, randrange
from math import tanh


class Perceptron:
 
    def __init__(self, number_of_inputs, label ):

        self.label = label
        self.number_of_inputs = number_of_inputs
        self.output = 0
        self.weights = np.array([random() for i in range(0, self.number_of_inputs) ])

    def learn(self, inputs, digit):

        # calculate output
        output = tanh(np.dot(inputs, self.weights))

        # calculate error
        # if the label fits the label of the node target = 1
        if self.label == digit:
            target = 1.0
        else:
            target = -1.0
        error = target - output 

        # update weights
        learning_rate = randrange(1,6)/100
        self.weights += learning_rate * inputs * error

    def predict(self, inputs):
        self.output = tanh(np.dot(inputs, self.weights))

        
        
