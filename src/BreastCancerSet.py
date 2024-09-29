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
        np.random.shuffle(self.data)

    def get_data(self):
        # this function returns only the data and no labels
        return self.data[:, :-1]

    def get_labels(self):
        # this function returns only the labels and no data
        return self.data[:, -1]

    def add_noise(self):

        # this function takes 10% of the amount of features and then shuffles all of that specific feature across all
        # classes. this function does not return anything it just updates the data variable

        # get the shape of the data and the amount of features to shuffle
        samples, features = np.shape(self.data[:, :-1])
        num_shuffled_features = int(features * .1 + 1)
        shuffled_cols = []

        # get the first column to shuffle
        curr_col = random.randint(0, features - 1)

        # shuffle the specified number of columns
        for i in range(num_shuffled_features):

            # make sure to not shuffle same column twice
            while curr_col in shuffled_cols:
                curr_col = random.randint(0, features - 1)

            feature_col = np.array(self.data[:, curr_col])
            np.random.shuffle(feature_col)

            self.data[:, curr_col] = feature_col


