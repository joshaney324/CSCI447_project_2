import numpy as np

from KNN import predict_regression, predict_classification, edited_nearest_neighbors_regression, predict_regression_forest
from Metric_functions import precision, recall, accuracy, mean_squared_error
import math as math

from src.Fold_functions import get_tune_folds, get_folds_regression
from src.KNN import edited_nearest_neighbors_regression
from src.ScikitMain import train_labels


def hyperparameter_tune_knn_classification(train_data, train_labels, test_data, test_labels, p_vals, k_vals):
    mean_squared_errors = [len(p_vals)][len(k_vals)]
    for p_val_index in range(len(p_vals)):
        for k_val_index in range(len(k_vals)):
            predictions = []
            for test_point in test_data:
                predictions.append(predict_classification(test_point, k_vals[k_val_index], p_vals[p_val_index]))
            predictions = np.array(predictions)
            mean_squared_errors[p_val_index][k_val_index] = mean_squared_error(test_labels, predictions, len(predictions))
    return mean_squared_errors
    #avg_metric = 0.0
    #k = None
    #p = None
    #for p_val in p_vals:
    #    for k_val in k_vals:
    #        predictions = []
    #        for test_point in test_data:
    #            predictions.append(predict_classification(train_data, train_labels, test_point, k_val, p_val))
    #        precisions = precision(predictions, test_labels)
    #        recalls = recall(predictions, test_labels)
    #        accuracies, matrix = accuracy(predictions, test_labels)
    #
    #        avg_precision = sum(precision_vals for _, precision_vals in precisions) / len(precisions)
    #        avg_recall = sum(recall_vals for _, recall_vals in recalls) / len(recalls)
    #        avg_accuracy = sum(accuracy_vals for _, accuracy_vals in accuracies) / len(accuracies)
    #
    #        avg_val = (avg_accuracy + avg_precision + avg_recall) / 3
    #
    #        if avg_metric < avg_val:
    #            avg_metric = avg_val
    #            k = k_val
    #            p = p_val
                # print("Best parameters so far")
                # print("Precision: " + str(avg_precision))
                # print("Recall: " + str(avg_recall))
                # print("Accuracy: " + str(avg_accuracy))
                # print("Average Metric: " + str(avg_val))
                # print("K: " + str(k))
                # print("P: " + str(p))

    #return k, p


def hyperparameter_tune_knn_regression(train_data, train_labels, test_data, test_labels, sigma_vals, p_vals, k_vals):
    mean_squared_errors = [len(sigma_vals)][len(p_vals)][len(k_vals)]
    # min_mean_squared_error = (0, 0, 0)
    # k = None
    # p = None
    # sigma = None
    for sigma_val_index in range(len(sigma_vals)):
        for p_val_index in range(len(p_vals)):
            for k_val_index in range(len(k_vals)):
                predictions = []
                for test_point in test_data:
                    if not forest:
                        predictions.append(predict_regression(train_data, train_labels, test_point, k_vals[k_val_index], p_vals[p_val_index], sigma_vals[sigma_val_index]))
                    else:
                        predictions.append(predict_regression_forest(train_data, train_labels, test_point, k_vals[k_val_index], p_vals[p_val_index], sigma_vals[sigma_val_index]))

                predictions = np.array(predictions)
                #mean_squared_val = mean_squared_error(test_labels, predictions, len(predictions))
                mean_squared_errors[sigma_val_index][p_val_index][k_val_index] = mean_squared_error(test_labels, predictions, len(predictions))
                #if mean_squared_val < min_mean_squared_error:
                #    min_mean_squared_error = mean_squared_val
                #    k = k_val
                #    p = p_val
                #    sigma = sigma_val
                #    print("Current Minimum Mean Squared Error: " + str(mean_squared_val))
                #    print("K: " + str(k) + "        P: " + str(p) + "        sigma: " + str(sigma))
    return mean_squared_errors


def hyperparameter_tune_edited_regression(train_data, train_labels, test_data, test_labels, threshold_vals, sigma_vals,  p_vals, k_vals, tolerance):
    mean_squared_errors = [len(threshold_vals)][len(sigma_vals)][len(p_vals)][len(k_vals)]
    for threshold_val_index in range(len(threshold_vals)):
        for sigma_val_index in range(len(sigma_vals)):
            edited_dataset = edited_nearest_neighbors_regression(train_data, train_labels, test_data, test_labels, threshold_vals[threshold_val_index], tolerance, sigma_vals[sigma_val_index])
            mean_squared_errors[threshold_val_index] = hyperparameter_tune_knn_regression(edited_dataset[:, :-1], edited_dataset[:, -1], test_data, test_labels, sigma_vals[sigma_val_index], p_vals, k_vals)

def cross_tune_edited_regression(data_folds, label_folds, threshold_vals, sigma_vals,  p_vals, k_vals, tolerance):
    mean_squared_errors_ave = np.zeros((len(threshold_vals), len(sigma_vals), len(p_vals), len(k_vals)))
    tune_data, tune_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
    train_data_folds, train_label_folds = get_folds_regression(train_data, train_labels, 10)
    for i in range(len(train_data_folds)):
        active_train_data = []
        active_train_labels = []
        for j in range(len(data_folds)):
            if i != j:
                for instance, label in zip(data_folds[j], label_folds[j]):
                    active_train_data.append(instance)
                    active_train_labels.append(label)
        active_train_data = np.array(active_train_data)
        active_train_labels = np.array(active_train_labels)
        mean_squared_errors_ave = mean_squared_errors_ave + hyperparameter_tune_edited_regression(active_train_data, active_train_labels, tune_data, tune_labels, threshold_vals, sigma_vals, p_vals, k_vals, tolerance)
    mean_squared_errors_ave = mean_squared_errors_ave / len(train_data_folds)
    min_mean_squared_error = sys.maxsize
    optimal = (0, 0, 0, 0)
    for threshold_val_index in range(len(threshold_vals)):
        for sigma_val_index in range(len(sigma_vals)):
            for p_val_index in range(len(p_vals)):
                for k_val_index in range(len(k_vals)):
                    if mean_squared_errors_ave[threshold_val_index][sigma_val_index][p_val_index][k_val_index] < min_mean_squared_error:
                        min_mean_squared_error = mean_squared_errors_ave[threshold_val_index][sigma_val_index][p_val_index][k_val_index]
                        optimal = (threshold_vals[threshold_val_index], sigma_vals[sigma_val_index], p_vals[p_val_index], k_vals[k_val_index])
    return optimal