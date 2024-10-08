from Metric_functions import precision, recall, accuracy, mean_squared_error
import numpy as np
from KNN import (predict_regression, predict_classification, edited_nearest_neighbors_classification,
                 edited_nearest_neighbors_regression, k_means_cluster)
from Fold_functions import get_folds_classification, get_folds_regression, get_tune_folds
from HyperparameterTune import hyperparameter_tune_knn_classification, hyperparameter_tune_knn_regression


def test_classification_dataset(dataset):
    data_folds, label_folds = get_folds_classification(dataset.get_data(), dataset.get_labels(), 10)
    test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
    k_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
    p_vals = [1, 2, 3, 4]
    k, p = hyperparameter_tune_knn_classification(train_data, train_labels, test_data, test_labels, k_vals, p_vals)
    print("Optimal hyperparameters")
    print("P: " + str(p) + "      K: " + str(k))
    data_folds, label_folds = get_folds_classification(train_data, train_labels, 10)
    cross_validate_classification(data_folds, label_folds, k, p)

    print("edited nearest neighbor")
    predictions = []
    edited_dataset = edited_nearest_neighbors_classification(train_data, train_labels, test_data, test_labels, 0.5)
    for data_point in test_data:
        predictions.append(predict_classification(edited_dataset[:, :-1], edited_dataset[:, -1], data_point, k, p))

    print(np.mean(predictions == test_labels))

    predictions = []
    print("k-means")
    centroids, centroid_labels = k_means_cluster(train_data, train_labels, int(len(edited_dataset)))

    # Print out centroids to see if they converged on each other
    # for centroid in centroids:
    #     print(centroid)

    for data_point in test_data:
        predictions.append(predict_classification(centroids, centroid_labels, data_point, k, p))

    print(np.mean(predictions == test_labels))


def test_regression_dataset(dataset):
    data_folds, label_folds = get_folds_regression(dataset.get_data(), dataset.get_labels(), 10)
    test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
    k_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
    p_vals = [1, 2]
    sigma_vals = [.05, .5, 1, 1.5, 2, 3, 4, 5]
    k, p, sigma = hyperparameter_tune_knn_regression(train_data, train_labels, test_data, test_labels, k_vals, p_vals,
                                                     sigma_vals)
    print("Optimal hyperparameters")
    print("P: " + str(p) + "      K: " + str(k) + "      Sigma: " + str(sigma))
    data_folds, label_folds = get_folds_regression(train_data, train_labels, 10)
    cross_validate_regression(data_folds, label_folds, k, p, sigma)

    print("edited nearest neighbor")
    predictions = []
    edited_dataset = edited_nearest_neighbors_regression(train_data, train_labels, test_data, test_labels, 1, 0.5, sigma)
    for data_point in test_data:
        predictions.append(predict_regression(edited_dataset[:, :-1], edited_dataset[:, -1], data_point, k, p, sigma))

    print(mean_squared_error(predictions, test_labels, len(predictions)))

    predictions = []
    print("k-means")
    centroids, centroid_labels = k_means_cluster(train_data, train_labels, int(len(edited_dataset)))

    # Print out centroids to see if they converged on each other
    # for centroid in centroids:
    #     print(centroid)

    for data_point in test_data:
        predictions.append(predict_regression(centroids, centroid_labels, data_point, k, p, sigma))

    print(mean_squared_error(predictions, test_labels, len(predictions)))


def binary_encoding(data, indices):
    uniques = [np.unique(data[:, col]) for col in indices]

    # create mapping from category to binary vectors
    category_to_binary = []
    for i in range(len(indices)):
        category_mapping = {}
        identity_matrix = np.eye(len(uniques[i]))

        for j, value in enumerate(uniques[i]):
            category_mapping[value] = identity_matrix[j]
        category_to_binary.append(category_mapping)

    binary_encoded_data = []

    # apply binary encoding
    for row in data:
        encoded_row = []
        for i, value in enumerate(row):
            if i in indices:
                # find corresponding binary vector and extend row
                col_index = indices.index(i)
                encoded_row.extend(category_to_binary[col_index][value])
            else:
                encoded_row.append(float(value))
        binary_encoded_data.append(encoded_row)
    return np.array(binary_encoded_data)


def cross_validate_classification(data_folds, label_folds, k_nearest_neighbors, p):
    # the cross_validate function is meant to get the precision, recall and accuracy values from each fold then print
    # out the average across folds. this function takes in a list of data folds and a list of label folds. it does not
    # return anything but prints out the metrics

    # Set up variables
    precision_avg = 0.0
    recall_avg = 0.0
    accuracy_avg = 0.0
    folds = len(data_folds)
    matrix_total = np.zeros((2,2))
    accuracies = []
    all_predictions = []
    all_labels = []

    # For each testing fold, set up a training and testing set and then append the loss function values
    for i in range(len(data_folds)):
        train_data = []
        test_data = []
        train_labels = []
        test_labels = []
        for j in range(len(data_folds)):
            if i != j:
                for instance, label in zip(data_folds[j], label_folds[j]):
                    train_data.append(instance)
                    train_labels.append(label)
            else:
                for instance, label in zip(data_folds[j], label_folds[j]):
                    test_data.append(instance)
                    test_labels.append(label)

        # make all the data into np arrays so that naive bayes class can use them
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        test_data = np.array(test_data)
        test_labels = np.array(test_labels)

        predictions = []
        for datapoint in test_data:
            predictions.append(predict_classification(train_data, train_labels, datapoint, k_nearest_neighbors, p))



        precision_vals = np.array(precision(predictions, test_labels))
        recall_vals = np.array(recall(predictions, test_labels))
        accuracy_vals, matrix = accuracy(predictions, test_labels)
        accuracy_vals = np.array(accuracy_vals)

        precision_total = 0
        recall_total = 0
        accuracy_total = 0
        counter = 0

        # get the averages of all the precision, recall, and accuracy values from all the folds
        for precision_val, recall_val, accuracy_val in zip(precision_vals, recall_vals, accuracy_vals):
            precision_total += float(precision_val[1])
            recall_total += float(recall_val[1])
            accuracies.append(float(accuracy_val[1]))
            accuracy_total += float(accuracy_val[1])
            matrix_total = matrix_total + np.array(matrix)
            counter += 1

        precision_avg += precision_total / counter
        recall_avg += recall_total / counter
        accuracy_avg += accuracy_total / counter

    print("Average precision: " + str(precision_avg / folds))
    print("Average recall: " + str(recall_avg / folds))
    print("Average accuracy: " + str(accuracy_avg / folds))

    return [precision_avg / folds, recall_avg / folds, accuracy_avg / folds], matrix_total, accuracies, all_predictions, all_labels


def cross_validate_regression(data_folds, label_folds, k_nearest_neighbors, p, sigma):
    # the cross_validate function is meant to get the precision, recall and accuracy values from each fold then print
    # out the average across folds. this function takes in a list of data folds and a list of label folds. it does not
    # return anything but prints out the metrics

    # Set up variables
    mean_squared_error_avg = 0.0
    folds = len(data_folds)

    # For each testing fold, set up a training and testing set and then append the loss function values
    for i in range(len(data_folds)):
        train_data = []
        test_data = []
        train_labels = []
        test_labels = []
        for j in range(len(data_folds)):
            if i != j:
                for instance, label in zip(data_folds[j], label_folds[j]):
                    train_data.append(instance)
                    train_labels.append(label)
            else:
                for instance, label in zip(data_folds[j], label_folds[j]):
                    test_data.append(instance)
                    test_labels.append(label)

        # make all the data into np arrays so that naive bayes class can use them
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        test_data = np.array(test_data)
        test_labels = np.array(test_labels)

        predictions = []
        for datapoint in test_data:
            predictions.append(predict_regression(train_data, train_labels, datapoint, k_nearest_neighbors, p, sigma))

        mean_squared_error_avg += mean_squared_error(test_labels, predictions, len(predictions))

    print("average mean squared error: " + str(mean_squared_error_avg / folds))
    return mean_squared_error_avg / folds




