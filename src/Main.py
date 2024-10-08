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
test_classification_dataset(breast_cancer)

# SOY BEAN
print("Soy Bean")
soy = SoyBeanSet()
test_classification_dataset(soy)

# GLASS
print("Glass")
glass = GlassSet(7)
test_classification_dataset(glass)

# ABALONE
print("Abalone")
abalone = AbaloneSet()
test_regression_dataset(abalone)

# FOREST FIRES
print("Forest")
forest = ForestFiresSet()
test_regression_dataset(forest)

# HARDWARE
print("Hardware")
machine = MachineSet()
test_regression_dataset(machine)
