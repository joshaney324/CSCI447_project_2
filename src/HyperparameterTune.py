from KNN import edited_nearest_neighbors_regression
from Fold_functions import get_folds_classification, get_folds_regression
from CrossValidateFunctions import (cross_validate_tune_classification, cross_validate_tune_regression,
                                    cross_validate_regression)
import math as math


def hyperparameter_tune_knn_classification(data_folds, label_folds, test_data, test_labels, k_vals, p_vals):

    # set up best value variables
    avg_metric = 0.0
    k = None
    p = None

    # for each value in k and p vals do a grid search
    for p_val in p_vals:
        for k_val in k_vals:

            # Run the tune specific cross validate function to test on the hold out tune fold
            avg_val = cross_validate_tune_classification(data_folds, label_folds, test_data, test_labels, k_val, p_val)

            # if the hyperparameters performed better update the best value variables
            if avg_metric < avg_val:
                avg_metric = avg_val
                k = k_val
                p = p_val

    return k, p


def hyperparameter_tune_knn_regression(data_folds, label_folds, test_data, test_labels, k_vals, p_vals, sigma_vals):

    # Set up best value variables
    min_mean_squared_error = math.inf
    k = None
    p = None
    sigma = None

    # For each value in p_vals, k_vals, and sigma_vals conduct a grid search to find best hyperparameters
    for p_val in p_vals:
        for k_val in k_vals:
            for sigma_val in sigma_vals:

                # Run the tune specific cross validate function to only test on the holdout fold
                mean_squared_val = cross_validate_tune_regression(data_folds, label_folds, test_data, test_labels, k_val, p_val, sigma_val)

                # If the mean squared error of the hyperparameters is the best, update the best value variables
                if mean_squared_val < min_mean_squared_error:
                    min_mean_squared_error = mean_squared_val
                    k = k_val
                    p = p_val
                    sigma = sigma_val
                    # print("Current Minimum Mean Squared Error: " + str(mean_squared_val))
                    # print("K: " + str(k) + "        P: " + str(p) + "        sigma: " + str(sigma))

    return k, p, sigma


def hyperparameter_tune_edited_regression(train_data, train_labels, test_data, test_labels, threshold_vals, sigma_vals):

    # Set up best value variabes
    error = None
    sigma = None
    min_mean_squared_error = math.inf

    # For each error value and sigma value conduct a grid search to figure out which performs the best
    for threshold_val_index in range(len(threshold_vals)):
        for sigma_val_index in range(len(sigma_vals)):
            # Sometimes with small error vals there are not enough datapoints to regress
            try:

                # get an edited dataset based off the error value and sigma value
                edited_dataset = edited_nearest_neighbors_regression(train_data, train_labels, test_data, test_labels, threshold_vals[threshold_val_index], sigma_vals[sigma_val_index])

                # Seperate data and labels
                edited_data = edited_dataset[:, :-1]
                edited_labels = edited_dataset[:, -1]

                # get folds off of the data and labels
                data_folds, label_folds = get_folds_regression(edited_data, edited_labels, 10)

                # Get the mean squared error value based off of the error and sigma value
                mean_squared_val = cross_validate_regression(data_folds, label_folds, 1, 2, sigma_vals[sigma_val_index])

                # If the model performed better update the best value variables
                if mean_squared_val < min_mean_squared_error:
                    error = threshold_vals[threshold_val_index]
                    sigma = sigma_vals[sigma_val_index]
                    min_mean_squared_error = mean_squared_val
                    # print("error: " + str(error))
                    # print("sigma: " + str(sigma))
                    # print("min_mean_squared_error: " + str(min_mean_squared_error))

            except:
                pass

    return error, sigma

