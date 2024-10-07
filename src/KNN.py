import math
import sys
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
def k_means_cluster(train_data, train_labels, num_clusters):
    # get number of features for each cluster and declare a 2d list cluster positions
    num_features = train_data.shape[1]
    centroids = np.empty((int(num_clusters), num_features))
    # generate starter values for each of the features between the minimum and maximum values for that feature
    for feature_index in range(centroids.shape[1]):
        max_val = np.max(train_data[:, feature_index])
        min_val = np.min(train_data[:, feature_index])
        for centroid_index in range(centroids.shape[0]):
            centroids[centroid_index][feature_index] = np.random.uniform(min_val, max_val)
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
            for centroid_index in range(centroids.shape[0]):
                if minkowski_metrics(train_data[entry_index], centroids[centroid_index], 2) < minkowski_metrics(train_data[entry_index], centroids[centroid_assignment], 2):
                    centroid_assignment = centroid_index
            centroid_assignments[entry_index] = centroid_assignment
        # find the average of all the points assigned to each centroid
        for centroid_index in range(centroids.shape[0]):
            # create an array to store the totals of all features of all assigned entries for the centroid
            centroid_ave = np.zeros(centroids.shape[1])
            counter = 0
            # add the features of each point assigned to the centroid to the total of all features for the centroid
            for entry_index in range(train_data.shape[0]):
                if centroid_assignments[entry_index] == centroid_index:
                    if counter > 0:
                        centroid_ave = centroid_ave + train_data[entry_index]
                        counter += 1
                    else:
                        centroid_ave = train_data[entry_index]
                        counter += 1
            # take the average features of all entries assigned to the cluster, or leave the cluster as is if no entries were assigned.
            if counter == 0:
                centroid_ave = centroids[centroid_index]
            else:
                centroid_ave = centroid_ave / counter
            # add the change in the centroid to the total of all changes in all centroids
            total_diff += minkowski_metrics(centroid_ave, centroids[centroid_index], 2)
            # reassign the centroid position
            centroids[centroid_index] = centroid_ave
        print(total_diff)
    # assign each centroid its nearest neighbor's class
    centroid_labels = np.empty(train_labels.shape)
    for centroid_index in range(centroids.shape[0]):
        min_distance = sys.maxsize
        for entry_index in range(train_labels.shape[0]):
            if minkowski_metrics(centroids[centroid_index], train_data[entry_index], 2) < min_distance:
                centroid_labels[centroid_index] = train_labels[entry_index]
                min_distance = minkowski_metrics(centroids[centroid_index], train_data[entry_index], 2)
    return centroids, centroid_labels


# def clustered_classification(train_data, train_labels, test_point, num_neighbors, p, num_clusters):
#     centroids, centroid_labels = k_means_cluster(train_data, train_labels, num_clusters)
#     return predict_classification(centroids, centroid_labels, test_point, num_neighbors, p)


# def clustered_regression(train_data, train_labels, test_point, num_neighbors, p, sigma, num_clusters):
#     centroids, centroid_labels = k_means_cluster(train_data, train_labels, num_clusters)
#     return predict_regression(centroids, centroid_labels, test_point, num_neighbors, p, sigma)


def edited_nearest_neighbors_classification(train_data, train_labels, test_data, test_labels, tolerance):
    # Force a shape on train labels so you can concatenate
    if train_labels.ndim == 1:
        train_labels = train_labels.reshape(-1, 1)

    # Concatenate data and labels
    edited_dataset = np.concatenate((train_data, train_labels), axis=1)

    # Set tracker variables for performance
    new_performance = 1.0
    old_performance = 0.0
    improved = True
    counter = 0

    while improved:
        # Create a new dataset to append all correct predictions to
        new_dataset = []

        # Create predictions list
        predictions = []

        # Test each data point in the dataset to see if it is predicted wrong
        for instance, label in zip(edited_dataset[:, :-1], edited_dataset[:, -1]):

            # Get the data point to remove for the training set
            removed_point = np.append(instance, label)
            remove_test_point_data = []

            # Remove the data point from the train set
            removed_point = False
            for data_point in edited_dataset:
                if not np.array_equal(data_point, removed_point) or removed_point:
                    remove_test_point_data.append(data_point)
                else:
                    removed_point = True

            # Convert to np array
            remove_test_point_data = np.array(remove_test_point_data)

            # Predict the label
            # Use the train set without the test instance
            # If the classifier gets it right add the point to the new dataset
            if predict_classification(remove_test_point_data[:, :-1], remove_test_point_data[:, -1], instance, 2, 2) == label:
                new_dataset.append(np.append(instance, label))

        # Convert the new dataset to an np array
        new_dataset = np.array(new_dataset)

        # Use the new dataset as the training data and test against the test set
        for instance in test_data:
            predictions.append(predict_classification(new_dataset[:, :-1], new_dataset[:, -1], instance, 1, 2))

        # Get a new performance
        new_performance = np.mean(predictions == test_labels)

        # if the performance has been worse for the past 6 times break out and return the dataset e
        if counter > 6:
            improved = False
        # if the counter has not gotten to 6 but the performance was worse increase the counter
        elif old_performance - new_performance > tolerance and counter < 6:
            counter += 1
        # else set the new performance
        else:
            if len(edited_dataset) == len(new_dataset):
                break

            edited_dataset = new_dataset
            old_performance = new_performance
            counter = 0
    return edited_dataset


def edited_nearest_neighbors_regression(train_data, train_labels, test_data, test_labels, error, tolerance, sigma):
    # Force a shape on train labels so you can concatenate
    if train_labels.ndim == 1:
        train_labels = train_labels.reshape(-1, 1)

    # Concatenate data and labels
    edited_dataset = np.concatenate((train_data, train_labels), axis=1)

    # Set tracker variables for performance
    new_performance = 1.0
    old_performance = 0.0
    improved = True
    counter = 0

    while improved:
        # Create a new dataset to append all correct predictions to
        new_dataset = []

        # Create predictions list
        predictions = []


        remove_test_point_data = []

        # Test each data point in the dataset to see if it is predicted wrong
        for instance, label in zip(edited_dataset[:, :-1], edited_dataset[:, -1]):

            # Get the data point to remove for the training set
            removed_point = np.append(instance, label)
            remove_test_point_data = []

            # Remove the data point from the train set
            for data_point in edited_dataset:
                if not np.array_equal(data_point, removed_point):
                    remove_test_point_data.append(data_point)

            # Predict the label using the training set without the test instance
            if abs(predict_regression(remove_test_point_data[:, :-1], remove_test_point_data[:, -1], instance, 2, 2, sigma) - label) <= error:
                new_dataset.append(np.append(instance, label))
        new_dataset = np.array(new_dataset)
        for instance in test_data:
            predictions.append(predict_regression(new_dataset[:, :-1], new_dataset[:, -1], instance, 1, 2, sigma))

        predictions = np.array(predictions)
        new_performance = mean_squared_error(predictions, test_labels, len(predictions))

        if new_performance - old_performance < tolerance and counter > 6:
            improved = False
        elif new_performance - old_performance < tolerance and counter < 6:
            counter += 1
        else:
            edited_dataset = new_dataset
            old_performance = new_performance
            counter = 0

    return edited_dataset
