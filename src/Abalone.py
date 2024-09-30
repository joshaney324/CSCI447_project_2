import csv
import numpy as np
import random as random
from HelperFunctions import binary_encoding

class AbaloneSet:
    def __init__(self):

        # read in the data from the csv file
        with open("../datasets/forestfires.data", "r") as data_file:
            self.data = list(csv.reader(data_file, delimiter=','))
        # skip header row
        self.data = self.data[1:]

        # convert data to a numpy array, remove extra row, and shuffle
        self.data = np.array(self.data[:-1])
        np.random.shuffle(self.data)

        # apply binary coding to categorical columns
        self.data = binary_encoding(self.data, [2, 3])

    def get_data(self):
        # return only data and no labels
        return self.data[:, :-1]

    def get_labels(self):
        # return only labels and no data
        return self.data[:, -1]