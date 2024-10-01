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
train_data = data[:600, :]
test_data = data[600:, :]
train_labels = labels[:600]
test_labels = labels[600:]

knn_classifier.fit(train_data, train_labels)
predictions = knn_classifier.predict(test_data)

precision_val = 0
recall_val = 0
accuracy_val = 0

precision_vals = np.array(precision(predictions, test_labels))
for val in precision_vals:
    precision_val += val[1]

recall_vals = np.array(recall(predictions, test_labels))
for val in recall_vals:
    recall_val += val[1]

accuracy_vals, matrix = accuracy(predictions, test_labels)
accuracy_vals = np.array(accuracy_vals)

for val in accuracy_vals:
    accuracy_val += val[1]

print("Breast: ")
print(precision_val / len(precision_vals))
print(recall_val / len(recall_vals))
print(accuracy_val / len(accuracy_vals))

data = soy.get_data()
labels = soy.get_labels()
train_data = data[:40, :]
test_data = data[40:, :]
train_labels = labels[:40]
test_labels = labels[40:]

knn_classifier.fit(train_data, train_labels)
predictions = knn_classifier.predict(test_data)

precision_val = 0
recall_val = 0
accuracy_val = 0

precision_vals = np.array(precision(predictions, test_labels))
for val in precision_vals:
    precision_val += float(val[1])

recall_vals = np.array(recall(predictions, test_labels))
for val in recall_vals:
    recall_val += float(val[1])

accuracy_vals, matrix = accuracy(predictions, test_labels)
accuracy_vals = np.array(accuracy_vals)

for val in accuracy_vals:
    accuracy_val += float(val[1])

print("Soy: ")
print(precision_val / len(precision_vals))
print(recall_val / len(recall_vals))
print(accuracy_val / len(accuracy_vals))

data = glass.get_data()
labels = glass.get_labels()
train_data = data[:200, :]
test_data = data[200:, :]
train_labels = labels[:200]
test_labels = labels[200:]

knn_classifier.fit(train_data, train_labels)
predictions = knn_classifier.predict(test_data)

precision_val = 0
recall_val = 0
accuracy_val = 0

precision_vals = np.array(precision(predictions, test_labels))
for val in precision_vals:
    precision_val += float(val[1])

recall_vals = np.array(recall(predictions, test_labels))
for val in recall_vals:
    recall_val += float(val[1])

accuracy_vals, matrix = accuracy(predictions, test_labels)
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
train_data = data[:4000, :]
test_data = data[4000:, :]
train_labels = labels[:4000]
test_labels = labels[4000:]

knn.fit(train_data, train_labels)
predictions = knn.predict(test_data)

mse = mean_squared_error(test_labels, predictions, len(predictions))
print("Abalone: " + str(mse))

data = forest.get_data()
labels = forest.get_labels()
train_data = data[:480, :]
test_data = data[480:, :]
train_labels = labels[:480]
test_labels = labels[480:]

knn.fit(train_data, train_labels)
predictions = knn.predict(test_data)

mse = mean_squared_error(test_labels, predictions, len(predictions))
print("Forest: " + str(mse))

data = machine.get_data()
labels = machine.get_labels()
train_data = data[:180, :]
test_data = data[180:, :]
train_labels = labels[:180]
test_labels = labels[180:]

knn.fit(train_data, train_labels)
predictions = knn.predict(test_data)

mse = mean_squared_error(test_labels, predictions, len(predictions))
print("Machine: " + str(mse))

# SKlearn model
# Breast:
# 0.9745710784313726
# 0.9745710784313726
# 0.9759036144578314
# Soy:
# 1.0
# 1.0
# 1.0
# Glass:
# 0.6388888888888888
# 0.38
# 0.7692307692307694
# Abalone: 4.653559322033896
# Forest: 30191.919243891894
# Machine: 3622.379259259259

# Our model
# breast metrics:
# 0.9666117986808646
# 0.9634253246753246
# 0.9675832892649392
# soy metrics:
# 1.0
# 1.0
# 1.0
# glass metrics:
# 0.7780119047619047
# 0.6632936507936508
# 0.8952687590187592
# abalone metric:
# 4.969036581638425
# forest metric:
# 5015.9253086957815
# machine metric:
# 3523.66776122034