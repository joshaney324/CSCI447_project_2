import math
from Metric_functions import precision, recall, accuracy, mean_squared_error
import numpy as np
from KNN import predict_regression, predict_classification


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


def get_folds_classification(data, labels, num_folds):

    # the get_folds function is meant to split the data up into a specified number of folds. this function takes in a
    # Dataset object as well as a specified number of folds. it then returns a list of all the data folds and label
    # folds



    # determine the number of instances of each class in each fold,
    # storing the values in a 2d numpy array (each row is a fold, each column is a class)
    classes, num_instances = np.unique(labels, return_counts=True)
    num_instances_perfold = np.zeros((num_folds, len(classes)), int)
    for i in range(len(num_instances_perfold[0])):
        for j in range(len(num_instances_perfold)):
            num_instances_perfold[j,i] = num_instances[i] // num_folds
        num_extra = num_instances[i] % num_folds
        for k in range(num_extra):
            num_instances_perfold[k,i] += 1

    # declare two lists of np arrays, each list entry representing a fold,
    # one list with data and one with labels
    label_folds = []
    for i in range(num_folds):
        label_folds.append(np.empty(shape=0))
    data_folds = []
    for i in range(num_folds):
        data_folds.append(np.empty(shape=(0, len(data[0]))))

    # iterate down the columns (classes) in the num_instances_perfold array,
    # then across the rows (folds) in the array,
    # then get the number of instances of that class in that fold,
    # then iterate through the labels to add them,
    # and remove the instances added to that fold from the data/labels classes to ensure uniqueness
    for i in range(len(num_instances_perfold[:,0])):
        for j in range(len(num_instances_perfold[i])):
            num_instances_infold = num_instances_perfold[i,j]
            k = 0
            while k < len(labels):
                if classes[j] == labels[k]:
                    label_folds[i] = np.append(label_folds[i], labels[k])
                    data_folds[i] = np.vstack((data_folds[i], data[k]))
                    data = np.delete(data, k, 0)
                    labels = np.delete(labels, k)
                    num_instances_infold -= 1
                    k -= 1
                if num_instances_infold == 0:
                    break
                k += 1
    # return a tuple of data_folds, label_folds
    return data_folds, label_folds


def get_folds_regression(data, labels, num_folds):
    return np.array_split(data, num_folds, 0), np.array_split(labels, num_folds)


def hyperparameter_tune_knn_classification(train_data, train_labels, test_data, test_labels, k_vals, p_vals):
    avg_metric = 0.0
    k = None
    p = None
    for p_val in p_vals:
        for k_val in k_vals:
            predictions = []
            for test_point in test_data:
                predictions.append(predict_classification(train_data, train_labels, test_point, k_val, p_val))
            precisions = precision(predictions, test_labels)
            recalls = recall(predictions, test_labels)
            accuracies, matrix = accuracy(predictions, test_labels)

            avg_precision = sum(precision_vals for _, precision_vals in precisions) / len(precisions)
            avg_recall = sum(recall_vals for _, recall_vals in recalls) / len(recalls)
            avg_accuracy = sum(accuracy_vals for _, accuracy_vals in accuracies) / len(accuracies)

            avg_val = (avg_accuracy + avg_precision + avg_recall) / 3

            if avg_metric < avg_val:
                avg_metric = avg_val
                k = k_val
                p = p_val
                print("Best parameters so far")
                print("Precision: " + str(avg_precision))
                print("Recall: " + str(avg_recall))
                print("Accuracy: " + str(avg_accuracy))
                print("Average Metric: " + str(avg_val))
                print("K: " + str(k))
                print("P: " + str(p))

    return k, p


def hyperparameter_tune_knn_regression(train_data, train_labels, test_data, test_labels, k_vals, p_vals, sigma_vals):
    min_mean_squared_error = math.inf
    k = None
    p = None
    sigma = None
    for p_val in p_vals:
        for k_val in k_vals:
            for sigma_val in sigma_vals:
                predictions = []
                try:
                    for test_point in test_data:
                        predictions.append(predict_regression(train_data, train_labels, test_point, k_val, p_val, sigma_val))
                    mean_squared_val = mean_squared_error(test_labels, predictions, len(predictions))
                    if mean_squared_val < min_mean_squared_error:
                        min_mean_squared_error = mean_squared_val
                        k = k_val
                        p = p_val
                        sigma = sigma_val
                        print("Current Minimum Mean Squared Error: " + str(mean_squared_val))
                        print("K: " + str(k) + "        P: " + str(p) + "        sigma: " + str(sigma))
                except ZeroDivisionError:
                    print("ZeroDivisionError")

    return k, p, sigma


def get_tune_folds(data_folds, label_folds):
    test_data = np.array(data_folds[-1])
    test_labels = np.array(label_folds[-1])
    train_data = []
    train_labels = []
    for j in range(len(data_folds) - 1):
        for instance, label in zip(data_folds[j], label_folds[j]):
            train_data.append(instance)
            train_labels.append(label)

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    return test_data, test_labels, train_data, train_labels
