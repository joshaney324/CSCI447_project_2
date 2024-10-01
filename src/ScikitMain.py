import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from BreastCancerSet import BreastCancerSet
from SoyBeanSet import SoyBeanSet
from GlassSet import GlassSet
from Abalone import AbaloneSet
from ForestFires import ForestFiresSet
from Hardware import MachineSet
from Metric_functions import mean_squared_error, recall, precision, accuracy
from sklearn.metrics import precision_score, recall_score, accuracy_score
from HelperFunctions import get_folds_classification, get_folds_regression

breast = BreastCancerSet()
soy = SoyBeanSet()
glass = GlassSet(7)

knn_classifier = KNeighborsClassifier()

data = breast.get_data()
labels = breast.get_labels()

knn_classifier.fit(data, labels)
predictions = knn_classifier.predict(data)

precision_val = 0
recall_val = 0
accuracy_val = 0

precision_vals = np.array(precision(predictions, labels))
for val in precision_vals:
    precision_val += val[1]

recall_vals = np.array(recall(predictions, labels))
for val in recall_vals:
    recall_val += val[1]

accuracy_vals, matrix = accuracy(predictions, labels)
accuracy_vals = np.array(accuracy_vals)

for val in accuracy_vals:
    accuracy_val += val[1]

print("Breast: ")
print(precision_val / len(precision_vals))
print(recall_val / len(recall_vals))
print(accuracy_val / len(accuracy_vals))

data = soy.get_data()
labels = soy.get_labels()

knn_classifier.fit(data, labels)
predictions = knn_classifier.predict(data)

precision_val = 0
recall_val = 0
accuracy_val = 0

precision_vals = np.array(precision(predictions, labels))
for val in precision_vals:
    precision_val += float(val[1])

recall_vals = np.array(recall(predictions, labels))
for val in recall_vals:
    recall_val += float(val[1])

accuracy_vals, matrix = accuracy(predictions, labels)
accuracy_vals = np.array(accuracy_vals)

for val in accuracy_vals:
    accuracy_val += float(val[1])

print("Soy: ")
print(precision_val / len(precision_vals))
print(recall_val / len(recall_vals))
print(accuracy_val / len(accuracy_vals))

data = glass.get_data()
labels = glass.get_labels()

knn_classifier.fit(data, labels)
predictions = knn_classifier.predict(data)

precision_val = 0
recall_val = 0
accuracy_val = 0

precision_vals = np.array(precision(predictions, labels))
for val in precision_vals:
    precision_val += float(val[1])

recall_vals = np.array(recall(predictions, labels))
for val in recall_vals:
    recall_val += float(val[1])

accuracy_vals, matrix = accuracy(predictions, labels)
accuracy_vals = np.array(accuracy_vals)

for val in accuracy_vals:
    accuracy_val += float(val[1])

print("Glass: ")
print(precision_val / len(precision_vals))
print(recall_val / len(recall_vals))
print(accuracy_val / len(accuracy_vals))


abalone = AbaloneSet()
forest = ForestFiresSet()
machine = MachineSet()

knn = KNeighborsRegressor()

data = abalone.get_data()
labels = abalone.get_labels()

knn.fit(data, labels)
predictions = knn.predict(data)

mse = mean_squared_error(labels, predictions, len(predictions))
print("Abalone: " + str(mse))

data = forest.get_data()
labels = forest.get_labels()

knn.fit(data, labels)
predictions = knn.predict(data)

mse = mean_squared_error(labels, predictions, len(predictions))
print("Forest: " + str(mse))

data = machine.get_data()
labels = machine.get_labels()

knn.fit(data, labels)
predictions = knn.predict(data)

mse = mean_squared_error(labels, predictions, len(predictions))
print("Machine: " + str(mse))

# SKlearn model
# Breast:
# 0.9770015683931439
# 0.9814966640280447
# 0.9809663250366032
# Soy:
# 1.0
# 1.0
# 1.0
# Glass:
# 0.7735770469466122
# 0.6872321893219726
# 0.92018779342723
# Abalone: 3.3306200622456443
# Forest: 2941.421123698259
# Machine: 1323.8442512077295

# Our model
# breast metrics:
# 0.9680075511129121
# 0.9691017316017316
# 0.9708090957165523
# soy metrics:
# 1.0
# 1.0
# 1.0
# glass metrics:
# 0.6275595238095237
# 0.46722222222222226
# 0.8160714285714287
# abalone metric:
# 4.7935364631889525
# forest metric:
# 3197.597234407766
# machine metric:
# nan



