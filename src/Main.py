from BreastCancerSet import BreastCancerSet
from SoyBeanSet import SoyBeanSet
from GlassSet import GlassSet
from Abalone import AbaloneSet
from ForestFires import ForestFiresSet
from Hardware import MachineSet
from HelperFunctions import test_classification_dataset, test_regression_dataset

# BREAST CANCER
print("Breast Cancer")
breast_cancer = BreastCancerSet()
test_classification_dataset(breast_cancer, 550)

# # SOY BEAN
# print("Soy Bean")
# soy = SoyBeanSet()
# # 38 from edited
# test_classification_dataset(soy, 30)
#
# # GLASS
# print("Glass")
# glass = GlassSet(7)
# test_classification_dataset(glass, 170)

# FOREST FIRES
print("Forest")
forest = ForestFiresSet()
test_regression_dataset(forest, 300)

# HARDWARE
print("Hardware")
machine = MachineSet()
test_regression_dataset(machine, 50)

# ABALONE
print("Abalone")
abalone = AbaloneSet()
test_regression_dataset(abalone)

