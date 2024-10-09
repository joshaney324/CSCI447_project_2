from Abalone import AbaloneSet
from ForestFires import ForestFiresSet
from Hardware import MachineSet
import matplotlib.pyplot as plt
import numpy as np

# Abalone

abalone = AbaloneSet()

labels = abalone.get_labels()
labels = np.array(labels)
plt.hist(labels)

plt.xlabel("Rings")
plt.ylabel("Counts")
plt.title('Histogram of Abalone Ring Values')
plt.savefig('C:/Users/josh.aney/OneDrive/Documents/CSCI447/ML_project2/plots/AbaloneHist.png')
plt.show()

# Forest

forest = ForestFiresSet()

labels = forest.get_labels()
labels = np.array(labels)
plt.hist(labels)

plt.xlabel("Area Burned in Hectare Acres")
plt.ylabel("Counts")
plt.title('Histogram of Area Burned in Hectare Acres')
plt.savefig('C:/Users/josh.aney/OneDrive/Documents/CSCI447/ML_project2/plots/ForestHist.png')
plt.show()

# Hardware

hardware = MachineSet()

labels = hardware.get_labels()
labels = np.array(labels)
plt.hist(labels)

plt.xlabel("Relative Performance")
plt.ylabel("Counts")
plt.title('Histogram of Relative Performance')
plt.savefig('C:/Users/josh.aney/OneDrive/Documents/CSCI447/ML_project2/plots/HardwareHist.png')
plt.show()

