import csv
import numpy as np
import random as random


class BreastCancerSet:
    def __init__(self):

        # collect data and labels from the csv file
        with open("../datasets/breast-cancer-wisconsin.data", "r") as data_file:
            self.data = list(csv.reader(data_file, delimiter=','))

        valid_rows = []

        for row in self.data:
            if all(value.isdigit() for value in row):
                valid_rows.append(row)

        for i in range(len(valid_rows)):
            del valid_rows[i][0]

        self.data = np.array(valid_rows, dtype=int)

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

        np.random.shuffle(self.data)

    def get_data(self):
        # this function returns only the data and no labels
        return self.data[:, :-1]

    def get_labels(self):
        # this function returns only the labels and no data
        return self.data[:, -1]
