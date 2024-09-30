import math

from KNN import predict_classification, predict_regression
from BreastCancerSet import BreastCancerSet
from SoyBeanSet import SoyBeanSet
from GlassSet import GlassSet
from Abalone import AbaloneSet
from ForestFires import ForestFiresSet
from Hardware import MachineSet
from Metric_functions import precision, recall, accuracy, mean_squared_error
from HelperFunctions import get_folds, cross_validate_classification
import numpy as np

# BREAST CANCER
breastCancer = BreastCancerSet()
data_folds, data_labels = get_folds(breastCancer, 10)
print("Breast Cancer")
cross_validate_classification(data_folds, data_labels, 6, 2)

# SOY BEAN
soy = SoyBeanSet()
data_folds, data_labels = get_folds(soy, 10)
print("Soy")
cross_validate_classification(data_folds, data_labels, 6, 2)

# GLASS
glass = GlassSet(7)
data_folds, data_labels = get_folds(glass, 10)
print("Glass")
cross_validate_classification(data_folds, data_labels, 6, 2)

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
    predictions.append(predict_regression(train_data, train_labels, instance, 3, 2, 40))

predictions = np.array(predictions)
test_labels = np.array(test_labels)
print("Machine")
print(mean_squared_error(test_labels, predictions, len(predictions)))
