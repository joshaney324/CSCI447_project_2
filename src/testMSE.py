import numpy as np

from Abalone import AbaloneSet
from ForestFires import ForestFiresSet
from Hardware import MachineSet
print("abalone")
abalone = AbaloneSet()
data = abalone.get_data()
labels = abalone.get_labels()

mean_label = np.mean(labels)
sum_dev = 0.0
for label in labels:
    sum_dev += (label - mean_label) ** 2

mean_dev = sum_dev / (len(labels) - 1)
print(mean_dev)
print("forest")
forest = ForestFiresSet()
data = forest.get_data()
labels = forest.get_labels()

mean_label = np.mean(labels)
sum_dev = 0.0
for label in labels:
    sum_dev += (label - mean_label) ** 2

mean_dev = sum_dev / (len(labels) - 1)
print(mean_dev)

print("machine")
machine = MachineSet()
data = machine.get_data()
labels = machine.get_labels()

mean_label = np.mean(labels)
sum_dev = 0.0
for label in labels:
    sum_dev += (label - mean_label) ** 2

mean_dev = sum_dev / (len(labels) - 1)
print(mean_dev)
