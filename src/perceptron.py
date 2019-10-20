"""This module provides the implementation of a 1-layer perceptron network

The perceptron is instantiated using the network class. The constructors
requires two dataset objects, a training set and a validation/prediction set. The
number number of nodes to be setup will be deduced from the training set. 
The network class provides functionality to train, test and predict, based on
the perceptron classifier algorithm 
"""
import numpy as np
from sys import exit
from random import random, randrange, shuffle
from math import tanh, floor


class Node:
    """This class represents a perceptron node

    The node is not instantiated directly by the user but by the network
    class which will based on the dataset deicde how many nodes are needed.
    A node can either train or predict. 

    Args:
        number_of_inputs: How many inputs and weights the node shall have
        label: The symbol for which the perceptron node will be trained
    """
 
    def __init__(self, number_of_inputs, label ):

        self.label = label
        self.number_of_inputs = number_of_inputs
        self.output = 0
        self.weights = np.array([random() for i in range(0, self.number_of_inputs) ])

    def learn(self, inputs, digit):
        """Method to execute a learning step with a given input vector and label

        Args:
            inputs: a vector of numeric input values.
            digit: the symbol that the input represents.
        
        Returns:
            No return valu provided. 
        """


        output = tanh(np.dot(inputs, self.weights))

        if self.label == digit:
            target = 1.0
        else:
            target = -1.0
        error = target - output 

        learning_rate = randrange(1,6)/100
        self.weights += learning_rate * inputs * error

    def predict(self, inputs):
        """This method uses the inputs and calculates an output value

        Args:
            inputs: a vector of numeric input values
        
        Returns:
            The calculated output based on inputs and current state of the nodes' weights.
        """
        self.output = tanh(np.dot(inputs, self.weights))



class Network:
    """This class represents a network of perceptron nodes loaded with data

    The user needs to provide two dataset objects, one for training/testing
    and another one for validatin/prediction. The network object exposes the 
    high-level functionality of training, testing and predicting data. The 
    network uses properties of the train/test dataset to determing the number 
    of nodes to setup. The prediction/validation dataset should have the 
    same number of inputs and corresponding labels as the train/test dataset.
    
    Args:
        dataset_training: The dataset used for training and testing.
        dataset_validation: The dataset used for prediction after training/testing.
    """

    def __init__(self, dataset_training, dataset_validation ):
    
        self.dataset = dataset_training
        self.data_validation = dataset_validation
        self.number_of_inputs = self.dataset.number_of_cols * self.dataset.number_of_rows
        self.number_of_nodes = len(self.dataset.digits)
        self.inputs = np.zeros(self.number_of_inputs)
        self.nodes = np.array([ Node(self.number_of_inputs, self.dataset.digits[i]) for i in range(0, self.number_of_nodes) ])


    def train(self):
        """Method to apply the learning algorithm to the perceptron network

        This method uses the training fraction of the train/test dataset
        and performs the learning algorithm. The method can be applied 
        multiple times. It will change the internal state of the perceptron 
        nodes, aka the weights.
        """

        for i in range(0, len(self.dataset.train_labels)):

            for j in range(0, self.number_of_nodes):

                inputs = self.dataset.train_images[i,:]
                train_digit = self.dataset.train_labels[i]
                self.nodes[j].learn(inputs, train_digit)


    def test(self):
        """Method that calculates a success rate based on the test data

        The method uses the test fraction of the train/test dataset. It
        does not modify the internal state of the perceptrons. 
        """
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
        """Method to predict output from input values of the validation/prediction dataset

        This method write the predicted value out to standard output. The sequence of 
        prediction will be according the sequence in the validation/prediction set. For most
        application cases, a dataset object with the non-randomized input sequence should be
        used. The method does not modify the inner state of the perceptrons and it can be
        run multiple times. Only the training fraction of will be used, hence it is advisable
        to use a dataset with the split fraction set to 1.
        """
        
        for i in range(0, self.data_validation.number_of_images ):
        
            for j in range(0, self.number_of_nodes):
        
                inputs = self.data_validation.train_images[i,:]
                self.nodes[j].predict(inputs)
            
            predicted = [ self.nodes[i].output for i in range(0, self.number_of_nodes ) ]
            predicted_index = predicted.index(max(predicted))
            predicted_digit = self.data_validation.digits[predicted_index]
            print(predicted_digit)


class Dataset:
    """Class to provide training, test and validation/prediction data to the perceptron network

    The configuration of the dataset is crucial for the correct setup and operation of the 
    perceptron network. Namely, the split parameter, the known labels and the ranomize property
    determine for which purpose a dataset can be used.

    Args:
        file_name: name of the text file that contains data.
        split_ratio: value from 0 to 1 that determines the split between training and test
        randomized: boolean that deterimines whether the sequence of inputs shall be randomized
    """
    
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
        """Method that imports the training and test labels from file

        Args:
            file_name: the filename of the same file that has been used to import training/test data
        """
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
