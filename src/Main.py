import math

from KNN import predict_classification, predict_regression
from BreastCancerSet import BreastCancerSet
from Abalone import AbaloneSet
from Metric_functions import precision, recall, accuracy, mean_squared_error
import numpy as np

breastCancer = BreastCancerSet()

train_data = breastCancer.get_data()
train_labels = breastCancer.get_labels()
train_data, test_data = train_data[:600], train_data[600:]
train_labels, test_labels = train_labels[:600], train_labels[600:]

predictions = []
for instance in test_data:
    predictions.append(predict_classification(train_data, train_labels, instance, 3, 2))

predictions = np.array(predictions)
test_labels = np.array(test_labels)
print(np.mean(test_labels == predictions))

abalone = AbaloneSet()

train_data = abalone.get_data()
train_labels = abalone.get_labels()
train_data, test_data = train_data[:4130], train_data[4131:]
train_labels, test_labels = train_labels[:4130], train_labels[4131:]

predictions = []
for instance in test_data:
    predictions.append(math.floor(predict_regression(train_data, train_labels, instance, 4, 2, 10) + 0.5))

predictions = np.array(predictions)
test_labels = np.array(test_labels)
print(mean_squared_error(test_labels, predictions, len(predictions)))

