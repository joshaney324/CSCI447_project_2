import csv
import numpy as np


class Abalone:
    def __init__(self):

        # collect data and labels from the csv file
        with open("../datasets/abalone.data", "r") as data_file:
            self.data = list(csv.reader(data_file, delimiter=','))

        # convert the self.data tuple into an np array
        self.data = np.array(self.data)

        # get the categorical column and get all the unique categories
        categorical_data = self.data[:, 0]
        categorical_values = np.unique(categorical_data)

        # create a zeros matrix of shape categorical_data x categorical_values
        encoded_matrix = np.zeros((len(categorical_data), len(categorical_values)))

        # for each category in categorical data create a category and an idx of that category
        for idx, category in enumerate(categorical_data):

            # np.where creates a boolean array of where the categorical values equal the category. The index is then for
            # getting the first element of the tuple which is the array of indices. The second index is for getting
            # first occurrence of the matching category. Basically in short in gets the correct index of the category
            # for the encoded matrix
            category_idx = np.where(categorical_values == category)[0][0]

            # for every index in the categorical_data input the correct index in the encoded matrix
            encoded_matrix[idx, category_idx] = 1

        # concatenate the old self.data with the new encoded matrix
        self.data = self.data[:, 1:]
        self.data = np.concatenate((encoded_matrix, self.data), axis=1)

        # cast the datatype of float on self.data
        self.data = np.array(self.data, dtype=float)

    def get_data(self):
        # this function returns only the data and no labels
        return self.data[:, :-1]

    def get_labels(self):
        # this function returns only the labels and no data
        return self.data[:, -1]



