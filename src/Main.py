import math
from KNN import predict_classification, predict_regression
from BreastCancerSet import BreastCancerSet
from Abalone import Abalone
from Metric_functions import precision, recall, accuracy, mean_squared_error
import numpy as np

breastCancer = BreastCancerSet()

train_data = breastCancer.get_data()
train_labels = breastCancer.get_labels()
train_data, test_data = train_data[:600], train_data[600:]
train_labels, test_labels = train_labels[:600], train_labels[600:]

predictions = []
for instance in test_data:
    predictions.append(predict_classification(train_data, train_labels, instance, 6, 2))

predictions = np.array(predictions)
test_labels = np.array(test_labels)
print(np.mean(test_labels == predictions))

abalone = Abalone()

train_data = abalone.get_data()
train_labels = abalone.get_labels()
train_data, test_data = train_data[:4100], train_data[4161:]
train_labels, test_labels = train_labels[:4100], train_labels[4161:]

predictions = []
for instance in test_data:
    predictions.append(math.floor(predict_regression(train_data, train_labels, instance, 6, 2, 5)) + 0.5)

print(mean_squared_error(test_labels, predictions, 1))

