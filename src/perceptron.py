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
    dataset = Dataset( training_images_file_name, 0.75 )
    dataset.loadLabels(training_labels_file_name)

    # setup perceptron
    perceptron = Perceptron( dataset )

    # train perceptron
    success_rate = 0
    while success_rate < 0.90:
        perceptron.train()
        success_rate = perceptron.test()

    print(success_rate)

if __name__ == "__main__":
    main()