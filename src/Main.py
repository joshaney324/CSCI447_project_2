import math

from KNN import predict_classification, predict_regression
from BreastCancerSet import BreastCancerSet
from SoyBeanSet import SoyBeanSet
from GlassSet import GlassSet
from Abalone import AbaloneSet
from ForestFires import ForestFiresSet
from Hardware import MachineSet
from Metric_functions import precision, recall, accuracy, mean_squared_error
import numpy as np

# BREAST CANCER
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
print("Breast Cancer")
print(np.mean(test_labels == predictions))

# SOY BEAN
soy = SoyBeanSet()
train_data = soy.get_data()
train_labels = soy.get_labels()
train_data, test_data = train_data[:40], train_data[41:]
train_labels, test_labels = train_labels[:40], train_labels[41:]

predictions = []
for instance in test_data:
    predictions.append(predict_classification(train_data, train_labels, instance, 3, 2))

predictions = np.array(predictions)
test_labels = np.array(test_labels)
print("Soy bean")
print(np.mean(test_labels == predictions))

# GLASS
glass = GlassSet(7)
train_data = glass.get_data()
train_labels = glass.get_labels()
train_data, test_data = train_data[:190], train_data[191:]
train_labels, test_labels = train_labels[:190], train_labels[191:]

predictions = []
for instance in test_data:
    predictions.append(predict_classification(train_data, train_labels, instance, 3, 2))

predictions = np.array(predictions)
test_labels = np.array(test_labels)
print("Glass")
print(np.mean(test_labels == predictions))

# ABALONE
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
print("Abalone")
print(mean_squared_error(test_labels, predictions, len(predictions)))

# FOREST FIRES
forest = ForestFiresSet()
train_data = forest.get_data()
train_labels = forest.get_labels()
train_data, test_data = train_data[:450], train_data[451:]
train_labels, test_labels = train_labels[:450], train_labels[451:]

predictions = []
for instance in test_data:
    predictions.append(predict_regression(train_data, train_labels, instance, 3, 2, 10))

predictions = np.array(predictions)
test_labels = np.array(test_labels)
print("Forest")
print(mean_squared_error(test_labels, predictions, len(predictions)))

# HARDWARE
machine = MachineSet()
train_data = machine.get_data()
train_labels = machine.get_labels()
train_data, test_data = train_data[:190], train_data[191:]
train_labels, test_labels = train_labels[:190], train_labels[191:]

predictions = []
for instance in test_data:
    predictions.append(predict_regression(train_data, train_labels, instance, 3, 2, 10))

predictions = np.array(predictions)
test_labels = np.array(test_labels)
print("Machine")
print(mean_squared_error(test_labels, predictions, len(predictions)))
