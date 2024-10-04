import csv
import numpy as np
from HelperFunctions import binary_encoding


class AbaloneSet:
    def __init__(self):

        # read in the data from the csv file
        with open("../datasets/abalone.data", "r") as data_file:
            self.data = list(csv.reader(data_file, delimiter=','))

        self.data = np.array(self.data)
        # apply binary coding to categorical columns
        self.data = binary_encoding(self.data, [0])
        self.data = np.array(self.data, dtype=float)
        np.random.shuffle(self.data)
        features = self.data[:, :-1]
        labels = self.data[:, -1:]

        # Normalize all the feature rows from 0 to 1
        features_min = features.min(axis=0)
        features_max = features.max(axis=0)

        normalized_features = (features - features_min) / (features_max - features_min)

        self.data = np.concatenate((normalized_features, labels), axis=1)

    def get_data(self):
        # return only data and no labels
        return self.data[:, :-1]

    def get_labels(self):
        # return only labels and no data
        return self.data[:, -1]