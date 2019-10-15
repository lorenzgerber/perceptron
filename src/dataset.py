from sys import exit
from math import floor
from random import shuffle
import numpy as np


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
