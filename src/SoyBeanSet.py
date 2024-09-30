import csv
import numpy as np
import random as random


class SoyBeanSet:
    def __init__(self):

        # read in the data from the csv file
        with open("../datasets/soybean-small.data", "r") as data_file:
            self.data = list(csv.reader(data_file, delimiter=','))

        # convert data to a numpy array, remove extra row, and shuffle
        self.data = np.array(self.data[:-1])
        np.random.shuffle(self.data)

    def get_data(self):
        # return only data and no labels
        return np.array(self.data[:, :-1], dtype=float)

    def get_labels(self):
        # return only labels and no data
        return self.data[:, -1]
