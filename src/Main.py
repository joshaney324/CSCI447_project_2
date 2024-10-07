import math

import numpy as np

from BreastCancerSet import BreastCancerSet
from SoyBeanSet import SoyBeanSet
from GlassSet import GlassSet
from Abalone import AbaloneSet
from ForestFires import ForestFiresSet
from Hardware import MachineSet
from KNN import (k_means_cluster, edited_nearest_neighbors_classification, edited_nearest_neighbors_regression,
                 predict_classification, predict_regression)
from HelperFunctions import (get_folds_classification, get_folds_regression, cross_validate_classification,
                             cross_validate_regression, hyperparameter_tune_knn_classification,
                             hyperparameter_tune_knn_regression, get_tune_folds)
from Metric_functions import mean_squared_error

# BREAST CANCER
breastCancer = BreastCancerSet()
data_folds, label_folds = get_folds_classification(breastCancer.get_data(), breastCancer.get_labels(), 10)
print("Breast Cancer")
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
k_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
p_vals = [1, 2, 3, 4]
k, p = hyperparameter_tune_knn_classification(train_data, train_labels, test_data, test_labels, k_vals, p_vals)
print("Optimal hyperparameters")
print("P: " + str(p) + "      K: " + str(k))
data_folds, label_folds = get_folds_classification(train_data, train_labels, 10)
breast_metrics, _, _, _, _ = cross_validate_classification(data_folds, label_folds, k, p)

print("edited nearest neighbor")
predictions = []
edited_dataset = edited_nearest_neighbors_classification(train_data, train_labels, test_data, test_labels, 0.5)
for data_point in test_data:
    predictions.append(predict_classification(edited_dataset[:, :-1], edited_dataset[:, -1], data_point, k, p))

print(np.mean(predictions == test_labels))

predictions = []
print("k-means")
centroids, centroid_labels = k_means_cluster(train_data, train_labels, int(len(edited_dataset) ** (1/2)))

# Print out centroids to see if they converged on each other
for centroid in centroids:
    print(centroid)

for data_point in test_data:
    predictions.append(predict_classification(centroids, centroid_labels, data_point, k, p))

print(np.mean(predictions == test_labels))

# SOY BEAN
soy = SoyBeanSet()
data_folds, label_folds = get_folds_classification(soy.get_data(), soy.get_labels(), 10)
print("Soy")
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
k_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
p_vals = [1, 2, 3, 4]
k, p = hyperparameter_tune_knn_classification(train_data, train_labels, test_data, test_labels, k_vals, p_vals)
print("Optimal hyperparameters")
print("P: " + str(p) + "      K: " + str(k))
data_folds, label_folds = get_folds_classification(train_data, train_labels, 10)
soy_metrics, _, _, _, _ = cross_validate_classification(data_folds, label_folds, k, p)

print("edited nearest neighbor")
predictions = []
edited_dataset = edited_nearest_neighbors_classification(train_data, train_labels, test_data, test_labels, 0.5)
for data_point in test_data:
    predictions.append(predict_classification(edited_dataset[:, :-1], edited_dataset[:, -1], data_point, k, p))

print(np.mean(predictions == test_labels))

predictions = []
print("k-means")
centroids, centroid_labels = k_means_cluster(train_data, train_labels, int(len(edited_dataset) ** (1/2)))

# Print out centroids to see if they converged on each other
for centroid in centroids:
    print(centroid)

for data_point in test_data:
    predictions.append(predict_classification(centroids, centroid_labels, data_point, k, p))

print(np.mean(predictions == test_labels))

# GLASS
glass = GlassSet(7)
data_folds, label_folds = get_folds_classification(glass.get_data(), glass.get_labels(), 10)
print("Glass")
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
k_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
p_vals = [1, 2, 3, 4]
k, p = hyperparameter_tune_knn_classification(train_data, train_labels, test_data, test_labels, k_vals, p_vals)
print("Optimal hyperparameters")
print("P: " + str(p) + "      K: " + str(k))
data_folds, label_folds = get_folds_classification(train_data, train_labels, 10)
glass_metrics, _, _, _, _ = cross_validate_classification(data_folds, label_folds, k, p)

print("edited nearest neighbor")
predictions = []
edited_dataset = edited_nearest_neighbors_classification(train_data, train_labels, test_data, test_labels, 0.5)
for data_point in test_data:
    predictions.append(predict_classification(edited_dataset[:, :-1], edited_dataset[:, -1], data_point, k, p))

print(np.mean(predictions == test_labels))

predictions = []
print("k-means")
centroids, centroid_labels = k_means_cluster(train_data, train_labels, int(len(edited_dataset) ** (1/2)))

# Print out centroids to see if they converged on each other
for centroid in centroids:
    print(centroid)

for data_point in test_data:
    predictions.append(predict_classification(centroids, centroid_labels, data_point, k, p))

print(np.mean(predictions == test_labels))

# ABALONE
abalone = AbaloneSet()
data_folds, label_folds = get_folds_regression(abalone.get_data(), abalone.get_labels(), 10)
print("Abalone")
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
k_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
p_vals = [1, 2]
sigma_vals = [.05, .5, 1, 1.5, 2, 5, 10, 50, 100]
k, p, sigma = hyperparameter_tune_knn_regression(train_data, train_labels, test_data, test_labels, k_vals, p_vals, sigma_vals)
print("Optimal hyperparameters")
print("P: " + str(p) + "      K: " + str(k) + "      Sigma: " + str(sigma))
data_folds, label_folds = get_folds_regression(train_data, train_labels, 10)
abalone_metrics = cross_validate_regression(data_folds, label_folds, k, p, sigma)

print("edited nearest neighbor")
predictions = []
edited_dataset = edited_nearest_neighbors_regression(train_data, train_labels, test_data, test_labels, 1, 0.5)
for data_point in test_data:
    predictions.append(predict_regression(edited_dataset[:, :-1], edited_dataset[:, -1], data_point, k, p, sigma))

print(mean_squared_error(predictions, test_labels, len(predictions)))

predictions = []
print("k-means")
centroids, centroid_labels = k_means_cluster(train_data, train_labels, int(len(edited_dataset) ** (1/2)))

# Print out centroids to see if they converged on each other
for centroid in centroids:
    print(centroid)

for data_point in test_data:
    predictions.append(predict_regression(centroids, centroid_labels, data_point, k, p, sigma))

print(mean_squared_error(predictions, test_labels, len(predictions)))



# FOREST FIRES
forest = ForestFiresSet()
data_folds, label_folds = get_folds_regression(forest.get_data(), forest.get_labels(), 10)
print("Forest")
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
k_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
p_vals = [1, 2]
sigma_vals = [0.01, 0.05, 0.1, 1, 5, 10, 20, 50, 100, 200]
k, p, sigma = hyperparameter_tune_knn_regression(train_data, train_labels, test_data, test_labels, k_vals, p_vals, sigma_vals)
print("Optimal hyperparameters")
print("P: " + str(p) + "      K: " + str(k) + "      Sigma: " + str(sigma))
data_folds, label_folds = get_folds_regression(train_data, train_labels, 10)
forest_metrics = cross_validate_regression(data_folds, label_folds, k, p, sigma)

print("edited nearest neighbor")
predictions = []
edited_dataset = edited_nearest_neighbors_regression(train_data, train_labels, test_data, test_labels, 1, 0.5)
for data_point in test_data:
    predictions.append(predict_regression(edited_dataset[:, :-1], edited_dataset[:, -1], data_point, k, p, sigma))

print(mean_squared_error(predictions, test_labels, len(predictions)))

predictions = []
print("k-means")
centroids, centroid_labels = k_means_cluster(train_data, train_labels, int(len(edited_dataset) ** (1/2)))

# Print out centroids to see if they converged on each other
for centroid in centroids:
    print(centroid)

for data_point in test_data:
    predictions.append(predict_regression(centroids, centroid_labels, data_point, k, p, sigma))

print(mean_squared_error(predictions, test_labels, len(predictions)))

# HARDWARE
machine = MachineSet()
data_folds, label_folds = get_folds_regression(machine.get_data(), machine.get_labels(), 10)
print("Machine")
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
k_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
p_vals = [1, 2]
sigma_vals = [0.05, 0.1, 0.5, 1, 2, 3, 4, 5, 10, 20, 50, 100]
k, p, sigma = hyperparameter_tune_knn_regression(train_data, train_labels, test_data, test_labels, k_vals, p_vals, sigma_vals)
print("Optimal hyperparameters")
print("P: " + str(p) + "      K: " + str(k) + "      Sigma: " + str(sigma))
data_folds, label_folds = get_folds_regression(train_data, train_labels, 10)
hardware_metrics = cross_validate_regression(data_folds, label_folds, k, p, sigma)

print("edited nearest neighbor")
predictions = []
edited_dataset = edited_nearest_neighbors_regression(train_data, train_labels, test_data, test_labels, 1, 0.5)
for data_point in test_data:
    predictions.append(predict_regression(edited_dataset[:, :-1], edited_dataset[:, -1], data_point, k, p, sigma))

print(mean_squared_error(predictions, test_labels, len(predictions)))

predictions = []
print("k-means")
centroids, centroid_labels = k_means_cluster(train_data, train_labels, int(len(edited_dataset) ** (1/2)))

# Print out centroids to see if they converged on each other
for centroid in centroids:
    print(centroid)

for data_point in test_data:
    predictions.append(predict_regression(centroids, centroid_labels, data_point, k, p, sigma))

print(mean_squared_error(predictions, test_labels, len(predictions)))

print("breast metrics:")
for metric in breast_metrics:
    print(metric)

print("soy metrics:")
for metric in soy_metrics:
    print(metric)

print("glass metrics:")
for metric in glass_metrics:
    print(metric)

print("abalone metric: ")
print(abalone_metrics)
print("forest metric: ")
print(forest_metrics)
print("machine metric: ")
print(hardware_metrics)
