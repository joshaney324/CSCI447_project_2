import csv
import numpy as np
import random as random


class GlassSet:

    # this constructor takes in num_bins as a parameter giving the number of bins we want to
    # separate the values into
    # it also takes the number of classes to classify. Either 2 or 7

    def __init__(self, num_classes):

        # read in data from csv file
        with open("../datasets/glass.data", "r") as data_file:
            self.data = list(csv.reader(data_file, delimiter=','))

        # check for invalid rows and remove them
        invalid_rows = []
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                try:
                    self.data[i][j] = float(self.data[i][j])
                except ValueError:
                    invalid_rows.append(i)
        for row in invalid_rows:
            del self.data[row]

        # remove iterator row and delete extra row at the end of data
        self.data = np.array(self.data[:-1])
        self.data = np.delete(self.data, 0, 1)

        features = self.data[:, :-1]
        labels = self.data[:, -1]
        features = np.array(features, dtype=float)
        labels = np.array(labels).reshape(-1, 1)

        # Normalize all the feature rows from 0 to 1
        features_min = features.min(axis=0)
        features_max = features.max(axis=0)

        normalized_features = (features - features_min) / (features_max - features_min)

        self.data = np.concatenate((normalized_features, labels), axis=1)
        self.data = np.array(self.data, dtype=float)

        # if only classifying 2 classes set original classes accordingly
        if num_classes == 2:
            for i in range(len(self.data)):
                if int(self.data[i, 9]) in [1, 2, 3, 4]:
                    self.data[i, 9] = 1
                elif int(self.data[i, 9]) in [5, 6, 7]:
                    self.data[i, 9] = 2
        self.data = np.array(self.data, dtype=float)
        np.random.shuffle(self.data)

    def get_data(self):
        # return only data and no labels
        return self.data[:, :-1]

    def get_labels(self):
        # return only labels and no data
        return self.data[:, -1]