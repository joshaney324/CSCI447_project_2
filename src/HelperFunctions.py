from sklearn.metrics import confusion_matrix
from NaiveBayes import NaiveBayes
from Metric_functions import precision, recall, accuracy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def test_dataset(dataset, data_name):
    data_folds, label_folds = get_folds(dataset, 10)
    # get the precision, recall and accuracy values
    ori_avgs, matrix_total, ori_accuracies, predictions, true_labels = cross_validate(data_folds, label_folds)
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    plot_label = data_name.replace('_', ' ').title()
    
    # depending on the dataset, edge cases are covered in which class labels do not represent
    # the actual names of the classes
    if(data_name == "breast_cancer"):
        labels = np.vectorize({"2": 'benign', "4": 'malignant'}.get)(np.unique(true_labels))  
    elif(data_name == "glass"):
        map = {
            "1": 'building_windows_float',
            "2": 'building_windows_non_float',
            "3": 'vehicle_windows_float',
            "4": 'vehicle_windows_non_float',
            "5": 'containers',
            "6": 'tableware',
            "7": 'headlamps',
        }
        uniques = ["%d" % number for number in np.unique((true_labels))]
        labels = np.vectorize(map.get)(uniques)
        labels = [s.replace('_', ' ') for s in labels]
        labels = [s.replace('-', ' ').title() for s in labels]
    else:
        labels = np.unique(true_labels)

    # plotting of the dataset's confusion matrix
    cm = confusion_matrix(predictions, true_labels, labels=np.unique(true_labels))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels)
    # Add labels and title
    plt.title(plot_label + ' Confusion Matrix')
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    plt.savefig("../output/" + data_name + "_dataset_matrix.svg", format='svg', dpi=1200, bbox_inches='tight')

    # adding noise to the given dataset
    noisy_dataset = dataset
    noisy_dataset.add_noise()

    print("With Noise")
    data_folds, label_folds = get_folds(noisy_dataset, 10)
    # get the precision, recall and accuracy values
    noisy_avgs, matrix_total, noisy_accuracies, predictions, true_labels = cross_validate(data_folds, label_folds)

    # plotting of the noisy dataset's confusion matrix
    cm = confusion_matrix(predictions, true_labels, labels=np.unique(true_labels))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels)

    # Add labels and title
    plt.title('Noisy ' + plot_label + ' Confusion Matrix')
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')


    plt.savefig("../output/noisy_" + data_name + "_dataset_matrix.svg", format='svg', dpi=1200, bbox_inches='tight')

    plt.figure(figsize=(8, 6))

    # Create the box plot
    plt.boxplot([ori_accuracies, noisy_accuracies], tick_labels=['Original Accuracies', 'Noisy Accuracies'])

    plt.title('Box Plot of Original and Noisy Accuracies (' + plot_label + ' Dataset)')
    plt.xlabel('Accuracy Types')
    plt.ylabel('Accuracy')

    plt.savefig("../output/" + data_name + "_dataset_boxplot.svg", format='svg', dpi=1200, bbox_inches='tight')

    return ori_avgs, noisy_avgs


def cross_validate(data_folds, label_folds):
    # the cross_validate function is meant to get the precision, recall and accuracy values from each fold then print
    # out the average across folds. this function takes in a list of data folds and a list of label folds. it does not
    # return anything but prints out the metrics

    # Set up variables
    precision_avg = 0.0
    recall_avg = 0.0
    accuracy_avg = 0.0
    folds = len(data_folds)
    matrix_total = np.zeros((2,2))
    accuracies = []
    all_predictions = []
    all_labels = []

    # For each testing fold, set up a training and testing set and then append the loss function values
    for i in range(len(data_folds)):
        train_data = []
        test_data = []
        train_labels = []
        test_labels = []
        for j in range(len(data_folds)):
            if i != j:
                for instance, label in zip(data_folds[j], label_folds[j]):
                    train_data.append(instance)
                    train_labels.append(label)
            else:
                for instance, label in zip(data_folds[j], label_folds[j]):
                    test_data.append(instance)
                    test_labels.append(label)

        # make all the data into np arrays so that naive bayes class can use them
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        test_data = np.array(test_data)
        test_labels = np.array(test_labels)
        naive = NaiveBayes()
        naive.set_probabilities(train_data, train_labels)
        predictions = naive.classify(test_data)
        all_predictions.extend(predictions)
        all_labels.extend(test_labels)

        precision_vals = np.array(precision(predictions, test_labels))
        recall_vals = np.array(recall(predictions, test_labels))
        accuracy_vals, matrix = accuracy(predictions, test_labels)
        accuracy_vals = np.array(accuracy_vals)

        precision_total = 0
        recall_total = 0
        accuracy_total = 0
        counter = 0

        # get the averages of all the precision, recall, and accuracy values from all the folds
        for precision_val, recall_val, accuracy_val in zip(precision_vals, recall_vals, accuracy_vals):
            precision_total += float(precision_val[1])
            recall_total += float(recall_val[1])
            accuracies.append(float(accuracy_val[1]))
            accuracy_total += float(accuracy_val[1])
            matrix_total = matrix_total + np.array(matrix)
            counter += 1
        
        precision_avg += precision_total / counter
        recall_avg += recall_total / counter
        accuracy_avg += accuracy_total / counter

        # print("Fold " + str(i + 1))
        # print("Precision: " + str(precision_total/counter))
        # print("Recall: " + str(recall_total/counter))
        # print("Accuracy: " + str(accuracy_total/counter))

    print("Average precision: " + str(precision_avg / folds))
    print("Average recall: " + str(recall_avg / folds))
    print("Average accuracy: " + str(accuracy_avg / folds))

    return [precision_avg / folds, recall_avg / folds, accuracy_avg / folds], matrix_total, accuracies, all_predictions, all_labels


def get_folds(dataset, num_folds):

    # the get_folds function is meant to split the data up into a specified number of folds. this function takes in a
    # Dataset object as well as a specified number of folds. it then returns a list of all the data folds and label
    # folds

    labels = dataset.get_labels()
    data = dataset.get_data()

    # determine the number of instances of each class in each fold,
    # storing the values in a 2d numpy array (each row is a fold, each column is a class)
    classes, num_instances = np.unique(labels, return_counts=True)
    num_instances_perfold = np.zeros((num_folds, len(classes)), int)
    for i in range(len(num_instances_perfold[0])):
        for j in range(len(num_instances_perfold)):
            num_instances_perfold[j,i] = num_instances[i] // num_folds
        num_extra = num_instances[i] % num_folds
        for k in range(num_extra):
            num_instances_perfold[k,i] += 1

    # declare two lists of np arrays, each list entry representing a fold,
    # one list with data and one with labels
    label_folds = []
    for i in range(num_folds):
        label_folds.append(np.empty(shape=0))
    data_folds = []
    for i in range(num_folds):
        data_folds.append(np.empty(shape=(0, len(data[0]))))

    # iterate down the columns (classes) in the num_instances_perfold array,
    # then across the rows (folds) in the array,
    # then get the number of instances of that class in that fold,
    # then iterate through the labels to add them,
    # and remove the instances added to that fold from the data/labels classes to ensure uniqueness
    for i in range(len(num_instances_perfold[:,0])):
        for j in range(len(num_instances_perfold[i])):
            num_instances_infold = num_instances_perfold[i,j]
            k = 0
            while k < len(labels):
                if classes[j] == labels[k]:
                    label_folds[i] = np.append(label_folds[i], labels[k])
                    data_folds[i] = np.vstack((data_folds[i], data[k]))
                    data = np.delete(data, k, 0)
                    labels = np.delete(labels, k)
                    num_instances_infold -= 1
                    k -= 1
                if num_instances_infold == 0:
                    break
                k += 1
    # return a tuple of data_folds, label_folds
    return data_folds, label_folds