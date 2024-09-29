from HelperFunctions import minkowski_metrics
import numpy as np


def predict_classification(train_data, train_labels, test_point, k_neighbors, p):
    distances = []

    for datapoint, label in zip(train_data, train_labels):
        distances.append((label, minkowski_metrics(datapoint, test_point, p)))

    distances.sort(key=lambda x: x[1])
    nearest_neighbors = distances[:k_neighbors]
    nearest_neighbors = np.array(nearest_neighbors)
    class_instances, count = np.unique(nearest_neighbors[:, 0], return_counts=True)

    max_count = 0
    prediction = None
    for i in range(len(class_instances)):
        if count[i] > max_count:
            max_count = count[i]
            prediction = class_instances[i]

    return prediction







