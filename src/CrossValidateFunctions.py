import numpy as np
from KNN import predict_regression, predict_classification
from Metric_functions import precision, recall, accuracy, mean_squared_error


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
    #
    # print("Average precision: " + str(precision_avg / folds))
    # print("Average recall: " + str(recall_avg / folds))
    # print("Average accuracy: " + str(accuracy_avg / folds))

    return [precision_avg / folds, recall_avg / folds, accuracy_avg / folds] # , matrix_total, accuracies, all_predictions, all_labels
    # return (precision_avg / folds + recall_avg / folds + accuracy_avg / folds) / 3


def cross_validate_regression(data_folds, label_folds, k_nearest_neighbors, p, sigma):

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

    # print("average mean squared error: " + str(mean_squared_error_avg / folds))
    return mean_squared_error_avg / folds


def cross_validate_tune_regression(data_folds, label_folds, test_data, test_labels, k_nearest_neighbors, p, sigma):

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

    # print("Average precision: " + str(precision_avg / folds))
    # print("Average recall: " + str(recall_avg / folds))
    # print("Average accuracy: " + str(accuracy_avg / folds))

    # return [precision_avg / folds, recall_avg / folds, accuracy_avg / folds], matrix_total, accuracies, all_predictions, all_labels

    return (precision_avg / folds + recall_avg / folds + accuracy_avg / folds) / 3

