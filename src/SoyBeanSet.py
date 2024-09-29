import csv
import numpy as np
import random as random


class SoyBeanSet:
    def __init__(self):

        # read in the data from the csv file
        with open("../data/soybean-small.data", "r") as data_file:
            self.data = list(csv.reader(data_file, delimiter=','))

        # convert data to a numpy array, remove extra row, and shuffle
        self.data = np.array(self.data[:-1])
        np.random.shuffle(self.data)

    def get_data(self):
        # return only data and no labels
        return self.data[:, :-1]

    def get_labels(self):
        # return only labels and no data
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


