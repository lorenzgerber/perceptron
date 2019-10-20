from perceptron import Dataset
from perceptron import Network
from time import time
import sys


def main():

    # Parsing and checking cli args
    if (len(sys.argv) != 4):
        print("Usage: digits.py training_images.txt training_labels.txt validation_images.txt")
        sys.exit(0)

    training_images_file_name = sys.argv[1]
    training_labels_file_name = sys.argv[2]
    validation_images_file_name = sys.argv[3]
    
    # reading data
    dataset_training = Dataset( training_images_file_name, 0.75, randomize=True )
    dataset_training.loadLabels(training_labels_file_name)
    dataset_validation = Dataset(validation_images_file_name, 1, randomize=False )

    # setup perceptron
    network = Network( dataset_training, dataset_validation )

    # train perceptron
    success_rate = 0
    start_time = time()
    while success_rate < 0.90:
        network.train()
        success_rate = network.test()
        if time() - start_time > 15:
            success_rate = 1

    # validate perceptron
    network.predict()

if __name__ == "__main__":
    main()