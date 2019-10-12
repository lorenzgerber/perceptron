from perceptron.dataset import Dataset
from perceptron.perceptron import Perceptron
import sys

def main():

    # Parsing and checking cli args
    if (len(sys.argv) != 3):
        print("Usage: perceptron.py training_images.txt training_labels.txt")

    training_images_file_name = sys.argv[1]
    training_labels_file_name = sys.argv[2]
    
    # reading data
    training_images = Dataset( training_images_file_name )
    for i in range(10):
        training_images.printImage( i )

    training_images.loadLabels(training_labels_file_name)


    # divide data


    # train perceptron


    # validate perceptron

if __name__ == "__main__":
    main()