import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from BreastCancerSet import BreastCancerSet
from SoyBeanSet import SoyBeanSet
from GlassSet import GlassSet
from matplotlib.lines import Line2D

# Breast
breast = BreastCancerSet()
data = breast.get_data()
labels = breast.get_labels()

pca = PCA(n_components=2)
principal_components = pca.fit_transform(data)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=labels, cmap='viridis')


legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Malignant',
           markerfacecolor='yellow', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Benign',
           markerfacecolor='purple', markersize=10)
]

plt.legend(handles=legend_elements, loc='best')
plt.title('PCA Plot of Breast Cancer Dataset')
plt.tight_layout()
plt.savefig('C:/Users/josh.aney/OneDrive/Documents/CSCI447/ML_project2/plots/BreastPCA.png')
plt.show()

# Soy
soy = SoyBeanSet()
data = soy.get_data()
labels = soy.get_labels()

pca = PCA(n_components=2)
principal_components = pca.fit_transform(data)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=labels, cmap='viridis')


legend_elements = [

    Line2D([0], [0], marker='o', color='w', label='d4',
           markerfacecolor='#FDE725FF', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='d3',
           markerfacecolor='#5EC962FF', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='D2',
           markerfacecolor='#3B528BFF', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='D1',
           markerfacecolor='#440154FF', markersize=10),
]

plt.legend(handles=legend_elements, loc='best')
plt.tight_layout()
plt.title('PCA Plot of Soybean Dataset')
plt.savefig('C:/Users/josh.aney/OneDrive/Documents/CSCI447/ML_project2/plots/SoyPCA.png')
plt.show()

# Glass
glass = GlassSet(7)
data = glass.get_data()
labels = glass.get_labels()

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data)

# Create the scatter plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=labels, cmap='viridis')


# Define the labels and get the viridis colormap
labels = np.array([1, 2, 3, 4, 5, 6, 7])
cmap = plt.cm.get_cmap('viridis')


# Create legend elements
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='1: Building Windows Float',
           markerfacecolor=cmap(0/6), markersize=10),
    Line2D([0], [0], marker='o', color='w', label='2: Building Windows Non-Float',
           markerfacecolor=cmap(1/6), markersize=10),
    Line2D([0], [0], marker='o', color='w', label='3: Vehicle Windows Float',
           markerfacecolor=cmap(2/6), markersize=10),
    Line2D([0], [0], marker='o', color='w', label='4: Vehicle Windows Non-Float',
           markerfacecolor=cmap(3/6), markersize=10),
    Line2D([0], [0], marker='o', color='w', label='5: Containers',
           markerfacecolor=cmap(4/6), markersize=10),
    Line2D([0], [0], marker='o', color='w', label='6: Tableware',
           markerfacecolor=cmap(5/6), markersize=10),
    Line2D([0], [0], marker='o', color='w', label='7: Headlamps',
           markerfacecolor=cmap(6/6), markersize=10)  # You can also use cmap(1) directly if cmap has 7 colors.
]

# Add the legend to the plot
plt.legend(handles=legend_elements, loc='best')

# Finalize and show the plot
plt.tight_layout()
plt.title('PCA Plot of Glass Dataset')
plt.savefig('C:/Users/josh.aney/OneDrive/Documents/CSCI447/ML_project2/plots/GlassPCA.png')
plt.show()
