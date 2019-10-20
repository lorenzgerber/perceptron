import numpy as np
from sys import exit
from random import random, randrange, shuffle
from math import tanh, floor


class Node:
 
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



class Network:

    def __init__(self, dataset_training, dataset_validation ):
    
        self.dataset = dataset_training
        self.data_validation = dataset_validation
        self.number_of_inputs = self.dataset.number_of_cols * self.dataset.number_of_rows
        self.number_of_nodes = len(self.dataset.digits)
        self.inputs = np.zeros(self.number_of_inputs)
        self.nodes = np.array([ Node(self.number_of_inputs, self.dataset.digits[i]) for i in range(0, self.number_of_nodes) ])


    def train(self):

        for i in range(0, len(self.dataset.train_labels)):

            for j in range(0, self.number_of_nodes):

                inputs = self.dataset.train_images[i,:]
                train_digit = self.dataset.train_labels[i]
                self.nodes[j].learn(inputs, train_digit)


    def test(self):
        event_counter = 0.0
        success_counter = 0.0 

        for i in range(0, len(self.dataset.test_labels)):
            
            for j in range(0, self.number_of_nodes):
            
                inputs = self.dataset.test_images[i,:]
                self.nodes[j].predict(inputs)

            predicted = [ self.nodes[i].output for i in range(0, self.number_of_nodes ) ]
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
        
            for j in range(0, self.number_of_nodes):
        
                inputs = self.data_validation.train_images[i,:]
                self.nodes[j].predict(inputs)
            
            predicted = [ self.nodes[i].output for i in range(0, self.number_of_nodes ) ]
            predicted_index = predicted.index(max(predicted))
            predicted_digit = self.data_validation.digits[predicted_index]
            print(predicted_digit)


class Dataset:
    
    def __init__( self, file_name, split_ratio, randomize=True ):
        f = open( file_name, "r")
        lines = f.readlines()
        f.close()

        self.randomized = randomize

        dimensions = str.split( lines[2] )
        self.number_of_rows = int(dimensions[1])
        self.number_of_cols = int(dimensions[2])
        self.number_of_images = int(dimensions[0])
        self.digits = [ int(i) for i in list(dimensions[3])]
        self.number_of_images_training = floor(self.number_of_images * split_ratio)
        self.number_of_images_test = self.number_of_images - self.number_of_images_training
        
        indices = [i for i in range(0,self.number_of_images)]
        
        if self.randomized:
            shuffle(indices)
        
        self.indices_train = indices[:(self.number_of_images_training)]
        self.indices_test = indices[self.number_of_images_training:]
        
        image_data = lines[3:]
        self.train_images = np.array([list(map(int, image_data[i].split())) for i in self.indices_train], dtype=float)
        self.train_images = np.divide(self.train_images, 1000.0)
        
        self.test_images = np.array([list(map(int, image_data[i].split())) for i in self.indices_test], dtype=float)
        self.test_images = np.divide(self.test_images, 1000.0)


    def loadLabels( self, file_name ):
        f = open( file_name, "r")
        lines = f.readlines()
        f.close

        dimensions = str.split( lines[2])

        digits = [ int( i ) for i in list( dimensions[1])]
        if len(set(self.digits).intersection(digits)) != len(digits):
            print("the digits of data and labelset don't coincide")
            exit(0)

        if int(dimensions[0]) != self.number_of_images:
            print("number of labels does not correspond to number of images")
            exit(0)

        labels = lines[3:]

        self.train_labels = [int(labels[i]) for i in self.indices_train]
        self.test_labels = [int(labels[i]) for i in self.indices_test]
