from KNN import predict_classification
from BreastCancerSet import BreastCancerSet
from Metric_functions import precision, recall, accuracy
import numpy as np

breastCancer = BreastCancerSet()

train_data = breastCancer.get_data()
train_labels = breastCancer.get_labels()
train_data, test_data = train_data[:600], train_data[600:]
train_labels, test_labels = train_labels[:600], train_labels[600:]

predictions = []
for instance in test_data:
    predictions.append(predict_classification(train_data, train_labels, instance, 2, 6))

predictions = np.array(predictions)
test_labels = np.array(test_labels)
print(np.mean(test_labels == predictions))

