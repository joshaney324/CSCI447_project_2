import math as math

# This function is meant to tune the classification KNN algorithms. It takes in datafolds, labels folds, a test/tune set
# it also takes a list of k vals and p vals to test in the grid search
def hyperparameter_tune_knn_classification(data_folds, label_folds, test_data, test_labels, k_vals, p_vals):
    from CrossValidateFunctions import (cross_validate_tune_classification)
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


# This function is meant to tune all of the regression knn models. It takes datafolds and label folds as well as a
# test/tune set. It also takes in a list of k vals, p vals, and sigma vals.
def hyperparameter_tune_knn_regression(data_folds, label_folds, test_data, test_labels, k_vals, p_vals, sigma_vals):
    from CrossValidateFunctions import (cross_validate_tune_regression)
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


# This function is meant to tune the edited knn regression. It tunes the error value and sigma value. It takes in
# datafolds, label folds, test/tune set, a list of error values, and a list of sigma values. It will return the optimal
# hyperparameters
def hyperparameter_tune_edited_regression(data_folds, label_folds, tune_data, tune_labels, threshold_vals, sigma_vals):
    from CrossValidateFunctions import cross_validate_edited_regression
    # Set up best value variabes
    error = None
    sigma = None
    min_mean_squared_error = math.inf

    # For each error value and sigma value conduct a grid search to figure out which performs the best
    for threshold_val_index in range(len(threshold_vals)):
        for sigma_val_index in range(len(sigma_vals)):
            # Sometimes with small error vals there are not enough datapoints to regress
            try:

                # Get the mean squared error value based off of the error and sigma value
                mean_squared_val = cross_validate_edited_regression(data_folds, label_folds, tune_data, tune_labels, sigma_vals[sigma_val_index], threshold_vals[threshold_val_index])

                # If the model performed better update the best value variables
                if mean_squared_val < min_mean_squared_error:
                    error = threshold_vals[threshold_val_index]
                    sigma = sigma_vals[sigma_val_index]
                    min_mean_squared_error = mean_squared_val

            except:
                pass

    return error, sigma

