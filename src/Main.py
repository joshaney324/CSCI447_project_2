from BreastCancerSet import BreastCancerSet
from SoyBeanSet import SoyBeanSet
from GlassSet import GlassSet
from Abalone import AbaloneSet
from ForestFires import ForestFiresSet
from Hardware import MachineSet
from HelperFunctions import (get_folds, cross_validate_classification, cross_validate_regression,
                             hyperparameter_tune_knn_classification, hyperparameter_tune_knn_regression, get_tune_folds)

# BREAST CANCER
breastCancer = BreastCancerSet()
data_folds, label_folds = get_folds(breastCancer.get_data(), breastCancer.get_labels(), 10)
print("Breast Cancer")
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
k_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10]
p_vals = [1, 2]
k, p = hyperparameter_tune_knn_classification(train_data, train_labels, test_data, test_labels, k_vals, p_vals)
print("Optimal hyperparameters")
print("P: " + str(p) + "      K: " + str(k))
data_folds, label_folds = get_folds(train_data, train_labels, 10)
cross_validate_classification(data_folds, label_folds, k, p)

# SOY BEAN
soy = SoyBeanSet()
data_folds, label_folds = get_folds(soy.get_data(), soy.get_labels(), 10)
print("Soy")
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
k_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10]
p_vals = [1, 2]
k, p = hyperparameter_tune_knn_classification(train_data, train_labels, test_data, test_labels, k_vals, p_vals)
print("Optimal hyperparameters")
print("P: " + str(p) + "      K: " + str(k))
data_folds, label_folds = get_folds(train_data, train_labels, 10)
cross_validate_classification(data_folds, label_folds, k, p)

# GLASS
glass = GlassSet(7)
data_folds, label_folds = get_folds(glass.get_data(), glass.get_labels(), 10)
print("Glass")
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
k_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10]
p_vals = [1, 2]
k, p = hyperparameter_tune_knn_classification(train_data, train_labels, test_data, test_labels, k_vals, p_vals)
print("Optimal hyperparameters")
print("P: " + str(p) + "      K: " + str(k))
data_folds, label_folds = get_folds(train_data, train_labels, 10)
cross_validate_classification(data_folds, label_folds, k, p)

# ABALONE
abalone = AbaloneSet()
data_folds, label_folds = get_folds(abalone.get_data(), abalone.get_labels(), 10)
print("Abalone")
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
k_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
p_vals = [1, 2]
sigma_vals = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
k, p, sigma = hyperparameter_tune_knn_regression(train_data, train_labels, test_data, test_labels, k_vals, p_vals, sigma_vals)
print("Optimal hyperparameters")
print("P: " + str(p) + "      K: " + str(k) + "      Sigma: " + str(sigma))
data_folds, label_folds = get_folds(train_data, train_labels, 10)
cross_validate_regression(data_folds, label_folds, k, p, sigma)

# FOREST FIRES
forest = ForestFiresSet()
data_folds, label_folds = get_folds(forest.get_data(), forest.get_labels(), 10)
print("Forest")
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
k_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
p_vals = [1, 2]
sigma_vals = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
k, p, sigma = hyperparameter_tune_knn_regression(train_data, train_labels, test_data, test_labels, k_vals, p_vals, sigma_vals)
print("Optimal hyperparameters")
print("P: " + str(p) + "      K: " + str(k) + "      Sigma: " + str(sigma))
data_folds, label_folds = get_folds(train_data, train_labels, 10)
cross_validate_regression(data_folds, label_folds, k, p, sigma)

# HARDWARE
machine = MachineSet()
data_folds, label_folds = get_folds(machine.get_data(), machine.get_labels(), 10)
print("Machine")
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
k_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
p_vals = [1, 2]
sigma_vals = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
k, p, sigma = hyperparameter_tune_knn_regression(train_data, train_labels, test_data, test_labels, k_vals, p_vals, sigma_vals)
print("Optimal hyperparameters")
print("P: " + str(p) + "      K: " + str(k) + "      Sigma: " + str(sigma))
data_folds, label_folds = get_folds(train_data, train_labels, 10)
cross_validate_regression(data_folds, label_folds, k, p, sigma)
