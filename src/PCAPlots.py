import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from BreastCancerSet import BreastCancerSet
from SoyBeanSet import SoyBeanSet
from GlassSet import GlassSet

# Breast
breast = BreastCancerSet()
data = breast.get_data()
labels = breast.get_labels()

pca = PCA(n_components=2)
principal_components = pca.fit_transform(data)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=labels, cmap='viridis')

plt.colorbar(scatter, label='Class')


plt.tight_layout()
plt.show()

# Soy
soy = SoyBeanSet()
data = soy.get_data()
labels = soy.get_labels()

pca = PCA(n_components=2)
principal_components = pca.fit_transform(data)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=labels, cmap='viridis')

plt.colorbar(scatter, label='Label')

plt.tight_layout()
plt.show()

# Glass
glass = GlassSet(7)
data = glass.get_data()
labels = glass.get_labels()

pca = PCA(n_components=2)
principal_components = pca.fit_transform(data)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=labels, cmap='viridis')

plt.colorbar(scatter, label='Label')

plt.tight_layout()
plt.show()
