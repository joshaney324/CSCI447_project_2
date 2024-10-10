from Fold_functions import get_folds_classification, get_folds_regression, get_tune_folds
from HyperparameterTune import hyperparameter_tune_edited_regression
from CrossValidateFunctions import *


# This function is meant to run all the algorithms on the classification sets. It performs everything we talk about in
# the paper
def test_classification_dataset(dataset, num_clusters):
    # Get folds from the dataset
    data_folds, label_folds = get_folds_classification(dataset.get_data(), dataset.get_labels(), 10)

    # Set up the tuning folds
    tune_data, tune_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)

    # Set up hyper parameter range
    k_vals = [2, 3, 4, 5, 10, 15]
    p_vals = [1, 2]

    # Tune the hyperparameters
    data_folds, label_folds = get_folds_classification(train_data, train_labels, 10)
    k, p = hyperparameter_tune_knn_classification(data_folds, label_folds, tune_data, tune_labels, k_vals, p_vals,)
    print("Optimal hyperparameters")
    print("P: " + str(p) + "      K: " + str(k))

    print("Metrics: " + str(cross_validate_classification(data_folds, label_folds, k, p)))

    # Run edited knn to reduce the dataset
    print("edited nearest neighbor")
    print("Metrics")
    print(cross_validate_edited_classification(data_folds, label_folds, tune_data, tune_labels))

    # Run k-means clustering to get clusters to use as the new dataset
    print("k-means")
    print("Metrics")
    print(str(cross_validate_k_means_classification(data_folds, label_folds, tune_data, tune_labels, num_clusters)))


# This function is meant to run all the algorithms on the regression sets. It performs everything we talk about in
# the paper
def test_regression_dataset(dataset, num_clusters):
    # Get folds from the dataset
    data_folds, label_folds = get_folds_regression(dataset.get_data(), dataset.get_labels(), 10)

    # Get tuning folds
    tune_data, tune_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)

    # Set up hyperparameter values
    k_vals = [1, 2, 3, 4, 5, 10, 15]
    p_vals = [1, 2]
    sigma_vals = [.05, .5, 1, 1.5, 2, 5]

    # Tune the values based off of the raw data
    data_folds, label_folds = get_folds_regression(train_data, train_labels, 10)
    k, p, sigma = hyperparameter_tune_knn_regression(data_folds, label_folds, tune_data, tune_labels, k_vals, p_vals,
                                                     sigma_vals)

    # Print the optimal parameters
    print("Optimal hyperparameters")
    print("P: " + str(p) + "      K: " + str(k) + "      Sigma: " + str(sigma))

    # Evaluate the model
    print("Mean Squared Error: " + str(cross_validate_regression(data_folds, label_folds, k, p, sigma)))

    # Get edited dataset
    print("edited nearest neighbor")
    print("Mean Squared Error")
    sigma, error = hyperparameter_tune_edited_regression(data_folds, label_folds, tune_data, tune_labels, [1, 3, 5, 10, 20, 40], [.1, .5, 1, 5])
    print(cross_validate_edited_regression(data_folds, label_folds, tune_data, tune_labels, error, sigma))

    # Get centroids from the raw data using k-means clustering
    print("k-means")
    print("Mean Squared Error")
    print(cross_validate_k_means_regression(data_folds, label_folds, tune_data, tune_labels, num_clusters))


# This function is meant to perform 1-hot coding on any categorical data. It will return a manipulated version of the
# data
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
