from Abalone import AbaloneSet
from ForestFires import ForestFiresSet
from BreastCancerSet import BreastCancerSet
from Fold_functions import get_folds_regression, get_tune_folds, get_folds_classification
from HyperparameterTune import (hyperparameter_tune_knn_regression, hyperparameter_tune_edited_regression,
                                hyperparameter_tune_knn_classification)
from CrossValidateFunctions import (cross_validate_edited_regression, cross_validate_regression,
                                    cross_validate_k_means_regression, cross_validate_k_means_classification,
                                    cross_validate_edited_classification, cross_validate_classification)
import numpy as np

# Set up Datasets
abalone = AbaloneSet()
data = abalone.get_data()
labels = abalone.get_labels()

# Split the data into 40 stratified folds and use 1 as smaller dataset
data_folds, label_folds = get_folds_regression(data, labels, 40)

# Take the first stratified fold as your new data
small_abalone_data = np.array(data_folds[0])
small_abalone_labels = np.array(label_folds[0])

# Get folds from the smaller dataset
data_folds, label_folds = get_folds_regression(small_abalone_data, small_abalone_labels, 10)

# Get tune folds
tune_data, tune_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)

# Hyperparameters Values
k_vals = [1, 2, 3, 4, 5, 10, 15]
p_vals = [1, 2]
sigma_vals = [.05, .5, 1, 1.5, 2, 5]

# Get new folds based off of train data
data_folds, label_folds = get_folds_regression(train_data, train_labels, 10)

# Get hyperparameter vals
k, p, sigma = hyperparameter_tune_knn_regression(data_folds, label_folds, tune_data, tune_labels, k_vals, p_vals,
                                                 sigma_vals)

# Print vals
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
print(cross_validate_k_means_regression(data_folds, label_folds, tune_data, tune_labels, 10))


breast_Cancer = BreastCancerSet()
# Get folds from the dataset
data_folds, label_folds = get_folds_classification(breast_Cancer.get_data(), breast_Cancer.get_labels(), 10)

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
print(str(cross_validate_k_means_classification(data_folds, label_folds, tune_data, tune_labels, 50)))