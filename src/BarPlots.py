import matplotlib.pyplot as plt
import numpy as np


def add_labels(bar_line, values):
    for bar in bar_line:
        height = bar.get_height()
        values.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}',
                ha='center', va='bottom')


labels = ['Precision', 'Recall', 'Accuracy']

########################################## CLASSIFICATION ################################################

# Breast Cancer

nearest_neighbor = [0.9660381871574873, 0.9645562770562771, 0.9676097303014279]
edited_nearest_neighbor = [0.9612292788512302, 0.9509307359307361, 0.959518773135907]
k_means = [0.9655455276430887, 0.9653625541125542, 0.9675568482284506]

locations = np.arange(len(labels))

width = 0.2

fig, sub = plt.subplots()

KNN = sub.bar(locations - width, nearest_neighbor, width, label='K Nearest Neighbor Classification')
ENN = sub.bar(locations, edited_nearest_neighbor, width, label='Edited Nearest Neighbor Classification')
KNN_K_means = sub.bar(locations + width, k_means, width, label='K Nearest Neighbor Classification Centroids from K-Means Clustering')

sub.set_xlabel('Metrics')
sub.set_title('Metrics From Different Models on Breast Cancer Dataset')
sub.set_xticks(locations)
sub.set_xticklabels(labels)

sub.set_ylim(0, 1.5)

sub.legend()

add_labels(KNN, sub)
add_labels(ENN, sub)
add_labels(KNN_K_means, sub)

plt.tight_layout()
plt.savefig('C:/Users/josh.aney/OneDrive/Documents/CSCI447/ML_project2/plots/BreastCancerBarPlot.png')
plt.show()

# Soy
nearest_neighbor = [1.0, 1.0, 1.0]
edited_nearest_neighbor = [1.0, 1.0, 1.0]
k_means = [0.9875, 0.9875, 0.99]

locations = np.arange(len(labels))

width = 0.2

fig, sub = plt.subplots()

KNN = sub.bar(locations - width, nearest_neighbor, width, label='K Nearest Neighbor Classification')
ENN = sub.bar(locations, edited_nearest_neighbor, width, label='Edited Nearest Neighbor Classification')
KNN_K_means = sub.bar(locations + width, k_means, width, label='K Nearest Neighbor Classification Centroids from K-Means Clustering')

sub.set_xlabel('Metrics')
sub.set_title('Metrics From Different Models on Soybean Dataset')
sub.set_xticks(locations)
sub.set_xticklabels(labels)

sub.set_ylim(0, 1.5)

sub.legend()

add_labels(KNN, sub)
add_labels(ENN, sub)
add_labels(KNN_K_means, sub)

plt.tight_layout()
plt.savefig('C:/Users/josh.aney/OneDrive/Documents/CSCI447/ML_project2/plots/SoyBarPlot.png')
plt.show()

# Glass

nearest_neighbor = [0.708387085137085, 0.5304365079365079, 0.8413369408369409]
edited_nearest_neighbor = [0.6764940476190476, 0.5396825396825398, 0.8507264911014911]
k_means = [0.6343482905982906, 0.4626984126984127, 0.7756601731601732]

locations = np.arange(len(labels))

width = 0.2

fig, sub = plt.subplots()

KNN = sub.bar(locations - width, nearest_neighbor, width, label='K Nearest Neighbor Classification')
ENN = sub.bar(locations , edited_nearest_neighbor, width, label='Edited Nearest Neighbor Classification')
KNN_K_means = sub.bar(locations + width, k_means, width, label='K Nearest Neighbor Classification Centroids from K-Means Clustering')

sub.set_xlabel('Metrics')
sub.set_title('Metrics From Different Models on Glass Dataset')
sub.set_xticks(locations)
sub.set_xticklabels(labels)

sub.set_ylim(0, 1.5)

sub.legend()

add_labels(KNN, sub)
add_labels(ENN, sub)
add_labels(KNN_K_means, sub)

plt.tight_layout()
plt.savefig('C:/Users/josh.aney/OneDrive/Documents/CSCI447/ML_project2/plots/GlassBarPlot.png')
plt.show()

###################################################### REGRESSION ###################################################

labels = ["Mean Squared Error"]

# Abalone

nearest_neighbor = [6.589548401155328 / 10.395265947347061]
edited_nearest_neighbor = [6.212742680907067 / 10.395265947347061]
k_means = [8.334849126492824 / 10.395265947347061]

locations = np.arange(len(labels))

width = 0.2

fig, sub = plt.subplots()

KNN = sub.bar(locations - width, nearest_neighbor, width, label='K Nearest Neighbor Classification')
ENN = sub.bar(locations, edited_nearest_neighbor, width, label='Edited Nearest Neighbor Classification')
KNN_K_means = sub.bar(locations + width, k_means, width, label='K Nearest Neighbor Classification Centroids from K-Means Clustering')

sub.set_xlabel('Metrics')
sub.set_title('Relative Mean Squared Error From Different Models on Abalone Dataset')
sub.set_xticks(locations)
sub.set_xticklabels(labels)

sub.set_ylim(0, 2)

sub.legend()

add_labels(KNN, sub)
add_labels(ENN, sub)
add_labels(KNN_K_means, sub)

plt.tight_layout()
plt.savefig('C:/Users/josh.aney/OneDrive/Documents/CSCI447/ML_project2/plots/AbaloneBarPlot.png')
plt.show()

# Hardware

nearest_neighbor = [3728.5419498900687 / 25866.524705557593]
edited_nearest_neighbor = [31827.131010909234 / 25866.524705557593]
k_means = [4627.8231035897425 / 25866.524705557593]

locations = np.arange(len(labels))

width = 0.2

fig, sub = plt.subplots()

KNN = sub.bar(locations - width, nearest_neighbor, width, label='K Nearest Neighbor Classification')
ENN = sub.bar(locations, edited_nearest_neighbor, width, label='Edited Nearest Neighbor Classification')
KNN_K_means = sub.bar(locations + width, k_means, width, label='K Nearest Neighbor Classification Centroids from K-Means Clustering')

sub.set_xlabel('Metrics')
sub.set_title('Relative Mean Squared Error From Different Models on Hardware Dataset')
sub.set_xticks(locations)
sub.set_xticklabels(labels)

sub.set_ylim(0, 2)

sub.legend()

add_labels(KNN, sub)
add_labels(ENN, sub)
add_labels(KNN_K_means, sub)

plt.tight_layout()
plt.savefig('C:/Users/josh.aney/OneDrive/Documents/CSCI447/ML_project2/plots/HardwareBarPlot.png')
plt.show()

# Forest

nearest_neighbor = [5283.0001691368425 / 4052.063224823412]
edited_nearest_neighbor = [4906.770218163989 / 4052.063224823412]
k_means = [4814.371607523724 / 4052.063224823412]

locations = np.arange(len(labels))

width = 0.2

fig, sub = plt.subplots()

KNN = sub.bar(locations - width, nearest_neighbor, width, label='K Nearest Neighbor Classification')
ENN = sub.bar(locations, edited_nearest_neighbor, width, label='Edited Nearest Neighbor Classification')
KNN_K_means = sub.bar(locations + width, k_means, width, label='K Nearest Neighbor Classification Centroids from K-Means Clustering')

sub.set_xlabel('Metrics')
sub.set_title('Relative Mean Squared Error From Different Models on Forest Fire Dataset')
sub.set_xticks(locations)
sub.set_xticklabels(labels)

sub.set_ylim(0, 2)

sub.legend()

add_labels(KNN, sub)
add_labels(ENN, sub)
add_labels(KNN_K_means, sub)

plt.tight_layout()
plt.savefig('C:/Users/josh.aney/OneDrive/Documents/CSCI447/ML_project2/plots/ForestBarPlot.png')
plt.show()
