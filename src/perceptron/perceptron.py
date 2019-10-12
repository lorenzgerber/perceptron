import numpy as np
from random import random
from math import tanh

class Perceptron:

    def __init__(self, dataset, learning_rate ):
    
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.number_of_inputs = self.dataset.number_of_cols * self.dataset.number_of_rows
        self.number_of_nodes = len(self.dataset.digits)

        print(self.number_of_inputs)
        print(self.number_of_nodes)

        self.inputs = np.zeros(self.number_of_inputs)
        random_weights = np.array([random() for i in range(0, self.number_of_inputs * self.number_of_nodes) ])
        self.weights = np.reshape(random_weights, (self.number_of_inputs, self.number_of_nodes))
        self.nodes = np.zeros(self.number_of_nodes)

    def train(self):

        for i in range(0, len(self.dataset.train_labels) - 1):

            for j in range(0, self.number_of_nodes - 1):
                inputs = self.dataset.train_images[i,:]
                weights = self.weights[:,j]
                self.nodes[j] = tanh(np.dot(inputs, weights))
                print(self.nodes[j])

                # calculate error
                # if the label fits the label of the node target = 1
                # else -> target = 0
                # target - nodes[j] = error 

                # update weights
                # weights = learning_rate * inputs * errro

            
            




