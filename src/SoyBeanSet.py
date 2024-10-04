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

        unique_classes = np.unique(self.data[:, -1])
        numerical_class_labels = {}
        for i, unique_class in enumerate(unique_classes):
            numerical_class_labels[unique_class] = i

        for i in range(len(self.data)):
            self.data[i, -1] = numerical_class_labels[self.data[i, -1]]

        self.data = np.array(self.data, dtype=float)

    def get_data(self):
        # return only data and no labels
        return np.array(self.data[:, :-1], dtype=float)

    def get_labels(self):
        # return only labels and no data
        return self.data[:, -1]
