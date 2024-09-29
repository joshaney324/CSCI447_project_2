import numpy as np


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

