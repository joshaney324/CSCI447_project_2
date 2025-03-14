from BreastCancerSet import BreastCancerSet
from SoyBeanSet import SoyBeanSet
from GlassSet import GlassSet
from Abalone import AbaloneSet
from ForestFires import ForestFiresSet
from Hardware import MachineSet
from HelperFunctions import test_classification_dataset, test_regression_dataset

# Main
# For each dataset it sets up the dataset and calls test_classification_dataset or test_regression_dataset which tests
# all the algorithms on the dataset

# BREAST CANCER
print("Breast Cancer")
breast_cancer = BreastCancerSet()
test_classification_dataset(breast_cancer, 550)

# SOY BEAN
print("Soy Bean")
soy = SoyBeanSet()
test_classification_dataset(soy, 30)

# GLASS
print("Glass")
glass = GlassSet(7)
test_classification_dataset(glass, 170)

# HARDWARE
print("Hardware")
machine = MachineSet()
test_regression_dataset(machine, 50)

# FOREST FIRES
print("Forest")
forest = ForestFiresSet()
test_regression_dataset(forest, 20)

# ABALONE
print("Abalone")
abalone = AbaloneSet()
test_regression_dataset(abalone, 40)

