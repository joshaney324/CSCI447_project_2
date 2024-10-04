import math
from random import random

from Metric_functions import minkowski_metrics, rbf_kernel, mean_squared_error
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
    weighted_average = np.sum(weights * k_nearest_distances[:, 0]) / (np.sum(weights))
    # if math.isnan(weighted_average):
    #     return 0
    # weighted_average = np.sum(k_nearest_distances[:, 1] * k_nearest_distances[:, 0]) / np.sum(k_nearest_distances[:, 1])

    # return the weighted average
    return weighted_average


# this function takes in a training dataset (only features) and a number of clusters, randomly assigns positions for
# cluster centroids, then adjusts those centroid positions according to the positions of the training data until
# the centroids no longer move
def k_means_cluster(train_data, num_clusters):
    # get number of features for each cluster and declare a 2d list cluster positions
    num_features = train_data.shape[1]
    centroids = np.empty((num_clusters, num_features))
    # generate starter values for each of the features between the minimum and maximum values for that feature
    for feature_index in range(centroids.shape[0]):
        max_val = np.max(train_data[:, feature_index])
        min_val = np.min(train_data[:, feature_index])
        for centroid_index in range(centroids.shape[1]):
            centroids[centroid_index][feature_index] = random.uniform(min_val, max_val)
    # create a variable to track the total distance between old and new centroids
    total_diff = 10
    # this while loop keeps reassigning entries to centroids and adjusting the centroids accordingly until the
    # centroids no longer move
    # run this loop as long as there was some change in the centroids in the last run
    while total_diff > 0:
        # reset total distance to zero
        total_diff = 0
        # create an array to store the centroid assignments of different entries
        centroid_assignments = np.empty(train_data.shape[0])
        # store the centroid assignment of the current entry
        centroid_assignment = 0
        # assign all entries to their closest centroid
        for entry_index in range(train_data.shape[0]):
            for centroid_index in range(centroids.shape[1]):
                if minkowski_metrics(train_data[entry_index], centroids[centroid_index], 2) < minkowski_metrics(train_data[entry_index], centroids[centroid_assignment], 2):
                    centroid_assignment = centroid_index
            centroid_assignments[entry_index] = centroid_assignment
        for centroid_index in range(centroids.shape[0]):
            centroid_ave = np.zeros(centroids.shape[1])
            counter = 0
            has_assigned_entries = False
            for entry_index in range(train_data.shape[0]):
                if centroid_assignments[entry_index] == centroid_index:
                    if has_assigned_entries:
                        centroid_ave += train_data[entry_index]
                        counter += 1
                    else:
                        centroid_ave = train_data[entry_index]
                        counter += 1
                        has_assigned_entries = True
            centroid_ave = centroid_ave / counter
            total_diff += minkowski_metrics(centroid_ave, centroids[centroid_index], 2)
            centroids[centroid_index] = centroid_ave
    return centroids


def edited_nearest_neighbors_classification(train_data, train_labels, tolerance):
    if train_labels.ndim == 1:
        train_labels = train_labels.reshape(-1, 1)
    edited_dataset = np.concatenate((train_data, train_labels), axis=1)
    new_performance = 1.0
    old_performance = 0.0
    improved = True

    while improved:
        new_dataset = []
        predictions = []
        for instance, label in zip(edited_dataset[:, :-1], edited_dataset[:, -1]):
            if predict_classification(edited_dataset[:, :-1], edited_dataset[:, -1], instance, 2, 2) == label:
                new_dataset.append(np.append(instance, label))
        new_dataset = np.array(new_dataset)
        for instance in train_data:
            predictions.append(predict_classification(new_dataset[:, :-1], new_dataset[:, -1], instance, 1, 2))

        predictions = np.array(predictions)
        new_performance = np.mean(predictions == train_labels.flatten())

        if new_performance - old_performance < tolerance:
            improved = False
        else:
            edited_dataset = new_dataset
            old_performance = new_performance

    return edited_dataset


def edited_nearest_neighbors_regression(train_data, train_labels, error, tolerance, sigma):
    if train_labels.ndim == 1:
        train_labels = train_labels.reshape(-1, 1)
    edited_dataset = np.concatenate((train_data, train_labels), axis=1)
    new_performance = 1.0
    old_performance = 0.0
    improved = True

    while improved:
        new_dataset = []
        predictions = []
        for instance, label in zip(edited_dataset[:, :-1], edited_dataset[:, -1]):
            if abs(predict_regression(edited_dataset[:, :-1], edited_dataset[:, -1], instance, 2, 2, sigma) - label) <= error:
                new_dataset.append(np.append(instance, label))
        new_dataset = np.array(new_dataset)
        for instance in train_data:
            predictions.append(predict_regression(new_dataset[:, :-1], new_dataset[:, -1], instance, 1, 2, sigma))

        predictions = np.array(predictions)
        new_performance = mean_squared_error(predictions, train_labels, len(predictions))

        if old_performance - new_performance < tolerance or old_performance < new_performance:
            improved = False
        else:
            edited_dataset = new_dataset
            old_performance = new_performance

    return edited_dataset
