import numpy as np
from KNN import (edited_nearest_neighbors_classification, edited_nearest_neighbors_regression, k_means_cluster)
from Fold_functions import get_folds_classification, get_folds_regression, get_tune_folds
from HyperparameterTune import (hyperparameter_tune_knn_classification, hyperparameter_tune_knn_regression,
                                hyperparameter_tune_edited_regression)
from CrossValidateFunctions import cross_validate_regression, cross_validate_classification


def test_classification_dataset(dataset):
    # Get folds from the dataset
    data_folds, label_folds = get_folds_classification(dataset.get_data(), dataset.get_labels(), 10)

    # Set up the tuning folds
    test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)

    # Set up hyper parameter range
    k_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
    p_vals = [1, 2, 3, 4]

    # Tune the hyperparameters
    data_folds, label_folds = get_folds_classification(train_data, train_labels, 10)
    k, p = hyperparameter_tune_knn_classification(data_folds, label_folds, test_data, test_labels, k_vals, p_vals,)
    print("Optimal hyperparameters")
    print("P: " + str(p) + "      K: " + str(k))

    print("Metrics: " + str(cross_validate_classification(data_folds, label_folds, k, p)))

    # Run edited knn to reduce the dataset
    print("edited nearest neighbor")
    edited_dataset = edited_nearest_neighbors_classification(train_data, train_labels, test_data, test_labels)

    # Tune the hyperparameters off the of edited dataset
    edited_data_folds, edited_label_folds = get_folds_classification(edited_dataset[:, :-1], edited_dataset[:, -1], 10)
    k, p = hyperparameter_tune_knn_classification(edited_data_folds, edited_label_folds, test_data, test_labels, k_vals, p_vals)

    # Evaluate function
    print("Metrics")
    print(str(cross_validate_classification(edited_data_folds, edited_label_folds, k, p)))

    # Run k-means clustering to get clusters to use as the new dataset
    print("k-means")

    # Get centroids and labels
    centroids, centroid_labels = k_means_cluster(train_data, train_labels, int(len(edited_dataset)))

    data_folds, label_folds = get_folds_classification(centroids, centroid_labels, 10)

    # Tune off the centroids and labels
    k, p = hyperparameter_tune_knn_classification(data_folds, label_folds, test_data, test_labels, k_vals, p_vals)

    # Evaluate model
    print("Metrics")
    print(str(cross_validate_classification(data_folds, label_folds, k, p)))


def test_regression_dataset(dataset):
    # Get folds from the dataset
    data_folds, label_folds = get_folds_regression(dataset.get_data(), dataset.get_labels(), 10)

    # Get tuning folds
    test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)

    # Set up hyperparameter values
    k_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
    p_vals = [1, 2]
    sigma_vals = [.05, .5, 1, 1.5, 2, 3, 4, 5]

    # Tune the values based off of the raw data
    data_folds, label_folds = get_folds_regression(train_data, train_labels, 10)
    k, p, sigma = hyperparameter_tune_knn_regression(data_folds, label_folds, test_data, test_labels, k_vals, p_vals,
                                                     sigma_vals)

    # Print the optimal parameters
    print("Optimal hyperparameters")
    print("P: " + str(p) + "      K: " + str(k) + "      Sigma: " + str(sigma))

    # Get folds separate to the hold out tune fold
    data_folds, label_folds = get_folds_regression(train_data, train_labels, 10)

    # Evaluate the model
    print("Mean Squared Error: " + str(cross_validate_regression(data_folds, label_folds, k, p, sigma)))

    # Get edited dataset
    print("edited nearest neighbor")

    # Tune the error value and sigma value
    error, sigma = hyperparameter_tune_edited_regression(train_data, train_labels, test_data, test_labels, [0.75, 1, 2, 5, 10, 20], sigma_vals)

    # Get edited dataset
    edited_dataset = edited_nearest_neighbors_regression(train_data, train_labels, test_data, test_labels, error, sigma)

    edited_data_folds, edited_label_folds = get_folds_regression(edited_dataset[:, :-1], edited_dataset[:, -1], 10)
    # Tune hyperparameters based off of the edited dataset
    k, p, sigma = hyperparameter_tune_knn_regression(edited_data_folds, edited_label_folds, test_data, test_labels, k_vals, p_vals, sigma_vals)

    # Evaluate model
    print("Mean Squared Error")
    print(cross_validate_regression(edited_data_folds, edited_label_folds, k, p, sigma))

    # Get centroids from the raw data using k-means clustering
    print("k-means")
    centroids, centroid_labels = k_means_cluster(train_data, train_labels, int(len(edited_dataset)))

    # Tune off of the clusters
    data_folds, label_folds = get_folds_regression(centroids, centroid_labels, 10)
    k, p, sigma = hyperparameter_tune_knn_regression(data_folds, label_folds, test_data, test_labels, k_vals,
                                                     p_vals, sigma_vals)

    # Evaluate model
    print("Mean Squared Error")
    print(cross_validate_regression(data_folds, label_folds, k, p, sigma))


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
