import numpy as np


def minkowski_metrics(initial_point, target_point, p):
    total = 0
    for feature_i, feature_t in zip(initial_point, target_point):
        total += abs(feature_i - feature_t) ** p

    return total ** (1/p)


def forest_distance(initial_point, target_point, p):

    month_vals = {0: "jan", 1: "feb", 2: "mar", 3: "apr", 4: "may", 5: "jun", 6: "jul", 7: "aug", 8: "sep", 9: "oct",
                  10: "nov", 11: "dec"}
    weekday_vals = {0: "sun", 1: "mon", 2: "tue", 3: "wed", 4: "thu", 5: "fri", 6: "sat"}

    non_categorical_init = np.delete(initial_point, [2, 3])
    non_categorical_target = np.delete(target_point, [2, 3])


    non_categorical_total = 0
    for i in range(len(non_categorical_init)):
        non_categorical_total += abs(non_categorical_target[i] - non_categorical_init[i]) ** p

    # Cyclic distance for months (mod 12)
    month_init = month_vals[initial_point[2]]
    month_target = month_vals[target_point[2]]
    month_distance = min(abs(month_init - month_target), 12 - abs(month_init - month_target)) ** p

    # Cyclic distance for weekdays (mod 7)
    weekday_init = weekday_vals[initial_point[3]]
    weekday_target = weekday_vals[target_point[3]]
    weekday_distance = min(abs(weekday_init - weekday_target), 7 - abs(weekday_init - weekday_target)) ** p

    # Total distance is the sum of non-categorical and categorical components
    total_distance = non_categorical_total + month_distance + weekday_distance

    return total_distance ** (1 / p)


def rbf_kernel(distance, sigma):
    return np.exp(- (distance ** 2) / (2 * sigma ** 2))


def mean_squared_error(predictions, true_vals, n):
    error = 0.0

    # square all the differences between the predictions and the true values
    for i in range(len(predictions)):
        error += abs(true_vals[i] - predictions[i]) ** 2
    # return the error divided by n
    return error / n


def precision(predictions, labels):

    # the precision function is meant to calculate the precision metric for a specific prediction set. it returns a
    # zipped list of the specific precision values for each class

    # turn parameters into numpy arrays and get unique classes
    labels = np.array(labels)
    predictions = np.array(predictions)
    classes = np.unique(labels)
    class_precisions = []

    # for each class in the prediction set calculate the number of true positives divided by the sum of true positives
    # and false positives. then append it to the list of all precision values
    for class_instance in classes:
        tp = 0
        fp = 0
        for prediction, label in zip(predictions, labels):
            if prediction == class_instance and prediction == label:
                tp += 1
            elif prediction == class_instance and prediction != label:
                fp += 1
        if tp + fp != 0:
            class_precisions.append(float(tp/(tp + fp)))

    return list(zip(classes, class_precisions))


def recall(predictions, labels):

    # the recall function is meant to calculate the recall metric for a specific prediction set. it returns a
    # zipped list of the specific recall values for each class

    # turn parameters into numpy arrays and get unique classes
    labels = np.array(labels)
    predictions = np.array(predictions)
    classes = np.unique(labels)
    class_recalls = []

    # for each class in the prediction set calculate the number of true positives divided by the sum of true positives
    # and false negatives. then append it to the list of all precision values
    for class_instance in classes:
        tp = 0
        fn = 0
        for prediction, label in zip(predictions, labels):
            if prediction == class_instance and prediction == label:
                tp += 1
            elif prediction != class_instance and class_instance == label:
                fn += 1
        if tp + fn != 0:
            class_recalls.append(float(tp / (tp + fn)))
        
    return list(zip(classes, class_recalls))


def accuracy(predictions, labels):

    # the accuracy function is meant to calculate the accuracy metric for a specific prediction set. it returns a
    # zipped list of the specific accuracy values for each class

    # turn parameters into numpy arrays and get unique classes
    labels = np.array(labels)
    predictions = np.array(predictions)
    classes = np.unique(labels)
    class_accuracies = []

    # for each class in the prediction set calculate the sum of true positives and true negatives divided by the sum of
    # true positives, true negatives, false positives, and false negatives. then append it to the list of all accuracy
    # values
    for class_instance in classes:
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for prediction, label in zip(predictions, labels):
            if prediction == class_instance:
                if class_instance == label:
                    tp += 1
                else:
                    fp += 1
            else:
                if class_instance == label:
                    fn += 1
                else:
                    tn += 1
        class_accuracies.append(float((tp + tn) / (tp + tn + fp + fn)))

    matrix = [[tp, fp],[fn, tn]]
    return list(zip(classes, class_accuracies)), matrix

