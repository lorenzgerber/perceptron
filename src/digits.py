from dataset import Dataset
from perceptron import Perceptron
from time import time
import sys


def main():

    # Parsing and checking cli args
    if (len(sys.argv) != 4):
        print("Usage: perceptron.py training_images.txt training_labels.txt validation_images.txt")
        sys.exit(0)

    training_images_file_name = sys.argv[1]
    training_labels_file_name = sys.argv[2]
    validation_images_file_name = sys.argv[3]
    
    # reading data
    dataset_training = Dataset( training_images_file_name, 0.75, True )
    dataset_training.loadLabels(training_labels_file_name)
    dataset_validation = Dataset(validation_images_file_name, 1, False )

    # setup perceptron
    perceptron = Perceptron( dataset_training, dataset_validation )

    # train perceptron
    success_rate = 0
    start_time = time()
    while success_rate < 0.90:
        perceptron.train()
        success_rate = perceptron.test()
        if time() - start_time > 15:
            success_rate = 1

    # validate perceptron
    perceptron.predict()

if __name__ == "__main__":
    main()