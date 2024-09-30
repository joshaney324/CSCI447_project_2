from HelperFunctions import minkowski_metrics, rbf_kernel
import numpy as np


# The point of this function is to get all the distances from the test point to the training point. The parameters are
# the training data, training labels, test point, and a p value for the Minkowski metric
def get_distances(train_data, train_labels, test_point, p):
    distances = []

    # append a list to the list of distances which contains the label and then the distance
    for datapoint, label in zip(train_data, train_labels):
        distances.append((label, minkowski_metrics(datapoint, test_point, p)))

    # sort the distances and return them
    distances.sort(key=lambda x: x[1])
    return distances


# This function will predict a label for a test point based on KNN. The parameters are the train data, train labels,
# test point, how many nearest neighbors to base off of, and a p value for the Minkowski metric
def predict_classification(train_data, train_labels, test_point, k_neighbors, p):
    # get the sorted list of distances from the get_distances function
    distances = get_distances(train_data, train_labels, test_point, p)

    # get the k nearest neighbors and turn it into a np array
    nearest_neighbors = distances[:k_neighbors]
    nearest_neighbors = np.array(nearest_neighbors)

    # get the unique classes of the neighbors and the counts of how many times they show up
    class_instances, count = np.unique(nearest_neighbors[:, 0], return_counts=True)

    # get the class that has the max count and return it
    max_count = 0
    prediction = None
    for i in range(len(class_instances)):
        if count[i] > max_count:
            max_count = count[i]
            prediction = class_instances[i]

    return prediction


# this function is meant to return a prediction for a continuous value. The parameters are the training data, training
# labels, test point, k nearest neighbors, p for the Minkowski metric, and sigma for the rbf kernel
def predict_regression(train_data, train_labels, test_point, k_neighbors, p, sigma):
    # get the distances and values from the get_distances() function
    distances = get_distances(train_data, train_labels, test_point, p)

    # get k nearest distances
    k_nearest_distances = distances[:k_neighbors]

    # get the weights for all the k nearest distances by using the rbf_kernel() function
    weights = []
    for i in range(len(k_nearest_distances)):
        weights.append(rbf_kernel(k_nearest_distances[i][1], sigma))

    # convert the weights and k_nearest_distances lists into np arrays
    weights = np.array(weights)
    k_nearest_distances = np.array(k_nearest_distances)

    # get the weighted average by multiplying the weights by the distances and divided it by the sum of the weights
    weighted_average = np.sum(weights * k_nearest_distances[:, 0]) / np.sum(weights)

    # return the weighted average
    return weighted_average









