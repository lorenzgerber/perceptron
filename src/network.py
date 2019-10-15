import numpy as np
from random import random, randrange
from math import tanh
from perceptron import Perceptron

class Network:

    def __init__(self, dataset_training, dataset_validation ):
    
        self.dataset = dataset_training
        self.data_validation = dataset_validation
        self.number_of_inputs = self.dataset.number_of_cols * self.dataset.number_of_rows
        self.number_of_perceptrons = len(self.dataset.digits)
        self.inputs = np.zeros(self.number_of_inputs)
        self.perceptrons = np.array([ Perceptron(self.number_of_inputs, self.dataset.digits[i]) for i in range(0, self.number_of_perceptrons) ])


    def train(self):

        for i in range(0, len(self.dataset.train_labels)):

            for j in range(0, self.number_of_perceptrons):

                inputs = self.dataset.train_images[i,:]
                train_digit = self.dataset.train_labels[i]
                self.perceptrons[j].learn(inputs, train_digit)


    def test(self):
        event_counter = 0.0
        success_counter = 0.0 

        for i in range(0, len(self.dataset.test_labels)):
            
            for j in range(0, self.number_of_perceptrons):
            
                inputs = self.dataset.test_images[i,:]
                self.perceptrons[j].predict(inputs)

            predicted = [ self.perceptrons[i].output for i in range(0, self.number_of_perceptrons ) ]
            predicted_index = predicted.index(max(predicted))
            current_digit = self.dataset.digits[predicted_index]
            test_digit = self.dataset.test_labels[i]

            if current_digit == test_digit:
                success_counter += 1.0
                event_counter += 1.0
            else:
                event_counter += 1.0
        
        return( 1 / event_counter * success_counter )

    def predict(self):
        
        for i in range(0, self.data_validation.number_of_images ):
        
            for j in range(0, self.number_of_perceptrons):
        
                inputs = self.data_validation.train_images[i,:]
                self.perceptrons[j].predict(inputs)
            
            predicted = [ self.perceptrons[i].output for i in range(0, self.number_of_perceptrons ) ]
            predicted_index = predicted.index(max(predicted))
            predicted_digit = self.data_validation.digits[predicted_index]
            print(predicted_digit)
