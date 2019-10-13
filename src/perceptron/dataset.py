from sys import exit
from math import floor
from random import shuffle
import numpy as np


class Dataset:
    
    def __init__( self, file_name, split ):
        f = open( file_name, "r")
        lines = f.readlines()
        f.close()

        dimensions = str.split( lines[2] )
        self.number_of_rows = int(dimensions[1])
        self.number_of_cols = int(dimensions[2])
        self.number_of_images = int(dimensions[0])
        self.digits = [ int(i) for i in list(dimensions[3])]
        self.image_data = lines[3:]

        self.number_of_images_training = floor(self.number_of_images * 0.75)
        self.number_of_images_test = self.number_of_images - self.number_of_images_training
        indices = [i for i in range(0,self.number_of_images)]
        shuffle(indices)
        self.indices_train = indices[:(self.number_of_images_training - 1)]
        self.indices_test = indices[self.number_of_images_training:]
        
        self.train_images = np.array([list(map(int, self.image_data[i].split())) for i in self.indices_train], dtype=float)
        self.train_images = np.divide(self.train_images, 1000.0)
        
        self.test_images = np.array([list(map(int, self.image_data[i].split())) for i in self.indices_test], dtype=float)
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

        self.labels = lines[3:]

        self.train_labels = [int(self.labels[i]) for i in self.indices_train]
        self.test_labels = [int(self.labels[i]) for i in self.indices_test]


    def printImage(self, index_of_image ):
        image = str.split(self.image_data[ index_of_image ])
        for i in range(0,self.number_of_rows):
            print( ''.join(map(str, image[((i * self.number_of_rows) + 0):((i * self.number_of_rows) + self.number_of_cols)] )))
