import numpy as np
from KNN import (predict_regression, predict_classification, edited_nearest_neighbors_classification,
                 edited_nearest_neighbors_regression, k_means_cluster)
from Metric_functions import precision, recall, accuracy, mean_squared_error
from HyperparameterTune import hyperparameter_tune_knn_classification, hyperparameter_tune_knn_regression
from Fold_functions import get_folds_classification, get_folds_regression


def cross_validate_classification(data_folds, label_folds, k_nearest_neighbors, p):
    # the cross_validate function is meant to get the precision, recall and accuracy values from each fold then print
    # out the average across folds. this function takes in a list of data folds and a list of label folds. it does not
    # return anything but prints out the metrics

    # Set up variables
    precision_avg = 0.0
    recall_avg = 0.0
    accuracy_avg = 0.0
    folds = len(data_folds)
    matrix_total = np.zeros((2, 2))
    accuracies = []

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

    return [precision_avg / folds, recall_avg / folds, accuracy_avg / folds]


def cross_validate_regression(data_folds, label_folds, k_nearest_neighbors, p, sigma):

    # This function is meant to cross validate the regression sets it returns a mean squared error

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

    return mean_squared_error_avg / folds


def cross_validate_tune_regression(data_folds, label_folds, test_data, test_labels, k_nearest_neighbors, p, sigma):

    # This function is meant to cross validate the regression sets but leave out the testing folds and use the tune
    # folds instead

    # Set up variables
    mean_squared_error_avg = 0.0
    folds = len(data_folds)

    # For each testing fold, set up a training and testing set and then append the loss function values
    for i in range(len(data_folds)):
        train_data = []
        train_labels = []
        for j in range(len(data_folds)):
            if i != j:
                for instance, label in zip(data_folds[j], label_folds[j]):
                    train_data.append(instance)
                    train_labels.append(label)

        # make all the data into np arrays and set up the test data and labels as the hold out fold
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        test_data = np.array(test_data)
        test_labels = np.array(test_labels)

        # get predictions and append them
        predictions = []
        for datapoint in test_data:
            predictions.append(predict_regression(train_data, train_labels, datapoint, k_nearest_neighbors, p, sigma))

        mean_squared_error_avg += mean_squared_error(test_labels, predictions, len(predictions))

    return mean_squared_error_avg / folds


def cross_validate_tune_classification(data_folds, label_folds, test_data, test_labels, k_nearest_neighbors, p):

    # This function is meant to cross validate the classification sets but leave out the testing folds and use the tune
    # folds instead

    # Set up variables
    precision_avg = 0.0
    recall_avg = 0.0
    accuracy_avg = 0.0
    folds = len(data_folds)
    matrix_total = np.zeros((2,2))
    accuracies = []

    # For each testing fold, set up a training and testing set and then append the loss function values
    for i in range(len(data_folds)):
        train_data = []
        train_labels = []
        for j in range(len(data_folds)):
            if i != j:
                for instance, label in zip(data_folds[j], label_folds[j]):
                    train_data.append(instance)
                    train_labels.append(label)

        # make all the data into np arrays and set the test data and labels as the hold out fold
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        test_data = np.array(test_data)
        test_labels = np.array(test_labels)

        predictions = []

        # Get all predictions
        for datapoint in test_data:
            predictions.append(predict_classification(train_data, train_labels, datapoint, k_nearest_neighbors, p))

        # Get all precision recall and accuracy vals
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

    return (precision_avg / folds + recall_avg / folds + accuracy_avg / folds) / 3


def cross_validate_edited_classification(data_folds, label_folds, tune_data, tune_labels):
    # the cross_validate edited function is meant to get the precision, recall and accuracy values from each fold then
    # print out the average across folds. this function takes in a list of data folds and a list of label folds. it
    # does not return anything but prints out the metrics

    # Set up variables
    k_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
    p_vals = [1, 2, 3, 4]
    precision_avg = 0.0
    recall_avg = 0.0
    accuracy_avg = 0.0
    folds = len(data_folds)
    matrix_total = np.zeros((2,2))
    accuracies = []

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

        # Get edited dataset then tune
        edited_data = edited_nearest_neighbors_classification(train_data, train_labels, tune_data, tune_labels)
        edited_folds, edited_labels = get_folds_classification(edited_data[:, :-1], edited_data[:, -1], 10)
        k, p = hyperparameter_tune_knn_classification(edited_folds, edited_labels, tune_data, tune_labels, k_vals, p_vals)

        # Get all predictions
        predictions = []
        for datapoint in test_data:
            predictions.append(predict_classification(edited_data[:, :-1], edited_data[:, -1], datapoint, k, p))

        # Get all precision, accuracy, and recall vals
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

    return [precision_avg / folds, recall_avg / folds, accuracy_avg / folds]


def cross_validate_k_means_classification(data_folds, label_folds, tune_data, tune_labels, num_clusters):
    # the cross_validate_k_means function is meant to get the precision, recall and accuracy values from each fold then
    # print out the average across folds. This tests and tunes the k means algorithm and the plain knn classification
    # algorithm

    # Set up variables
    k_vals = [2, 3, 4, 5, 10, 15]
    p_vals = [1, 2]
    precision_avg = 0.0
    recall_avg = 0.0
    accuracy_avg = 0.0
    folds = len(data_folds)
    matrix_total = np.zeros((2, 2))
    accuracies = []

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

        # Get centroids and tune the knn model
        centroids, centroid_labels = k_means_cluster(train_data, train_labels, num_clusters)
        centroid_folds, centroid_label_folds = get_folds_classification(centroids, centroid_labels, 10)
        k, p = hyperparameter_tune_knn_classification(centroid_folds, centroid_label_folds, tune_data, tune_labels, k_vals,
                                                      p_vals)

        # Get all predictions
        predictions = []
        for datapoint in test_data:
            predictions.append(predict_classification(centroids, centroid_labels, datapoint, k, p))

        # Get the precision, recall, and accuracy values for the final hyperparameters
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

    return [precision_avg / folds, recall_avg / folds, accuracy_avg / folds]



def cross_validate_edited_regression(data_folds, label_folds, tune_data, tune_labels, error, sigma_edit):
    # This function is meant to edit, tune, and crossvalidate the edited knn regression model. It will return the mean
    # squared error

    k_vals = [2, 3, 4, 5, 10, 15]
    p_vals = [1, 2]
    sigma_vals = [0.1, 0.5, 1, 5]
    mean_squared_error_total = 0.0

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

        # edit the data, split into folds, and tune
        edited_data = edited_nearest_neighbors_regression(train_data, train_labels, tune_data, tune_labels, error, sigma_edit)
        edited_data_folds, edited_label_folds = get_folds_regression(edited_data[:, :-1], edited_data[:, -1], 10)
        k, p, sigma = hyperparameter_tune_knn_regression(edited_data_folds, edited_label_folds, tune_data, tune_labels
                                                         , k_vals, p_vals, sigma_vals)
        # Get all predictions
        predictions = []
        for datapoint in test_data:
            predictions.append(predict_regression(edited_data[:, :-1], edited_data[:, -1], datapoint, k, p, sigma))

        mean_squared_error_total += mean_squared_error(test_labels, predictions, len(predictions))

    return mean_squared_error_total / len(data_folds)


def cross_validate_k_means_regression(data_folds, label_folds, tune_data, tune_labels, num_clusters):
    # This functino is meant to get the centroids of a set of datafolds, tune, and crossvalidate. It returns a mean
    # squared error value at the end

    k_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
    p_vals = [1, 2, 3, 4]
    sigma_vals = [.05, .5, 1, 1.5, 2, 3, 4, 5]
    mean_squared_error_total = 0.0

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

        # get centroids, centroid folds and tune off of the new train set
        centroids, centroid_labels = k_means_cluster(train_data, train_labels, num_clusters)
        centroid_data_folds, centroid_label_folds = get_folds_regression(centroids, centroid_labels, 10)
        k, p, sigma = hyperparameter_tune_knn_regression(centroid_data_folds, centroid_label_folds, tune_data, tune_labels
                                                         , k_vals, p_vals, sigma_vals)

        # get all predictions and get the final mean squared error
        predictions = []
        for datapoint in test_data:
            predictions.append(predict_regression(centroids, centroid_labels, datapoint, k, p, sigma))

        mean_squared_error_total += mean_squared_error(test_labels, predictions, len(predictions))

    return mean_squared_error_total / len(data_folds)

