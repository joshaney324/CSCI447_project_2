import csv
import numpy as np
import random as random


class GlassSet:

    # this constructor takes in num_bins as a parameter giving the number of bins we want to
    # separate the values into
    # it also takes the number of classes to classify. Either 2 or 7

    def __init__(self, num_bins, num_classes):

        # read in data from csv file
        with open("../data/glass.data", "r") as data_file:
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

        # if only classifying 2 classes set original classes accordingly
        if num_classes == 2:
            for i in range(len(self.data)):
                if int(self.data[i, 9]) in [1, 2, 3, 4]:
                    self.data[i, 9] = 1
                elif int(self.data[i, 9]) in [5, 6, 7]:
                    self.data[i, 9] = 2

        # use number of bins from constructor to bin data accordingly
        for i in range(len(self.data[0]) - 1):
            column = self.data[:,i]
            # bins = np.histogram_bin_edges(column, bins = "auto")
            bins = [0]
            for j in range(num_bins):
                bins.append(np.percentile(column, j*(100/num_bins)))
            bin_assignments = np.digitize(column, bins)
            for j in range(len(column)):
                self.data[j][i] = bin_assignments[j]
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


