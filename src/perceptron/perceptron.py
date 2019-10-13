import numpy as np
from random import random, randrange
from math import tanh

class Perceptron:

    def __init__(self, dataset_training, dataset_validation ):
    
        self.dataset = dataset_training
        self.data_validation = dataset_validation
        self.number_of_inputs = self.dataset.number_of_cols * self.dataset.number_of_rows
        self.number_of_nodes = len(self.dataset.digits)

        self.inputs = np.zeros(self.number_of_inputs)
        random_weights = np.array([random() for i in range(0, self.number_of_inputs * self.number_of_nodes) ])
        self.weights = np.reshape(random_weights, (self.number_of_inputs, self.number_of_nodes))
        self.nodes = np.zeros(self.number_of_nodes)

    def train(self):

        for i in range(0, len(self.dataset.train_labels)):

            for j in range(0, self.number_of_nodes):
                inputs = self.dataset.train_images[i,:]
                weights = self.weights[:,j]
                self.nodes[j] = tanh(np.dot(inputs, weights))

                # calculate error
                # if the label fits the label of the node target = 1
                current_digit = self.dataset.digits[j]
                train_digit = self.dataset.train_labels[i]
                if current_digit == train_digit:
                    target = 1.0
                else:
                    target = -1.0
                error = target - self.nodes[j] 

                # update weights
                learning_rate = randrange(1,5)/100
                self.weights[:,j] += learning_rate * inputs * error


    def test(self):
        event_counter = 0.0
        success_counter = 0.0 
        for i in range(0, len(self.dataset.test_labels)):
            
            for j in range(0, self.number_of_nodes):
                inputs = self.dataset.test_images[i,:]
                weights = self.weights[:,j]
                self.nodes[j] = tanh(np.dot(inputs, weights))
        
            predicted_index = np.where(self.nodes == np.amax(self.nodes))[0][0]
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
            for j in range(0, self.number_of_nodes):
                inputs = self.data_validation.train_images[i,:]
                weights = self.weights[:,j]
                self.nodes[j] = tanh(np.dot(inputs, weights))
            
            predicted_index = np.where(self.nodes == np.amax(self.nodes))[0][0]
            predicted_digit = self.data_validation.digits[predicted_index]
            print(predicted_digit)






            
            




