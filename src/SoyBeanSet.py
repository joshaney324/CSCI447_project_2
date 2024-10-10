import csv
import numpy as np
import random as random


class SoyBeanSet:
    def __init__(self):

        # read in the data from the csv file
        with open("../datasets/soybean-small.data", "r") as data_file:
            self.data = list(csv.reader(data_file, delimiter=','))

        # convert data to a numpy array
        self.data = np.array(self.data)
        np.random.shuffle(self.data)

        # Enumerate classes and replace the string labes
        unique_classes = np.unique(self.data[:, -1])
        numerical_class_labels = {}
        for i, unique_class in enumerate(unique_classes):
            numerical_class_labels[unique_class] = i

        for i in range(len(self.data)):
            self.data[i, -1] = numerical_class_labels[self.data[i, -1]]

        # Set up features and labels
        features = self.data[:, :-1]
        labels = self.data[:, -1]
        features = np.array(features, dtype=float)
        labels = np.array(labels).reshape(-1, 1)

        # Delete Unnecessary
        features = np.delete(features, [10, 12, 13, 14, 15, 16, 17, 18, 28, 29, 30, 31, 32, 33], axis=1)

        # Normalize all the feature rows from 0 to 1
        features_min = features.min(axis=0)
        features_max = features.max(axis=0)

        normalized_features = (features - features_min) / (features_max - features_min)

        self.data = np.concatenate((normalized_features, labels), axis=1)
        self.data = np.array(self.data, dtype=float)

    def get_data(self):
        # return only data and no labels
        return np.array(self.data[:, :-1], dtype=float)

    def get_labels(self):
        # return only labels and no data
        return self.data[:, -1]
