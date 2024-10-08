import numpy as np


def get_tune_folds(data_folds, label_folds):
    test_data = np.array(data_folds[-1])
    test_labels = np.array(label_folds[-1])
    train_data = []
    train_labels = []
    for j in range(len(data_folds) - 1):
        for instance, label in zip(data_folds[j], label_folds[j]):
            train_data.append(instance)
            train_labels.append(label)

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    return test_data, test_labels, train_data, train_labels


def get_folds_regression(data, labels, num_folds):

    # Turn the data and labels into np arrays
    data = np.array(data)
    labels = np.array(labels)

    # Make sure it's the right shape
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    # Concatenate
    dataset = np.concatenate((data, labels), axis=1)

    # get a list of sorted indices and put the dataset into the correct order
    sorted_indices = np.argsort(dataset[:, -1])
    dataset = dataset[sorted_indices, :]

    # get groups of 10 consecutive data points and append it to the list of groups
    datapoint_groups = []
    for i in range(0, len(dataset), 10):
        group = dataset[i:i + 10]
        datapoint_groups.append(group)

    # Create 10 empty folds
    folds = [[] for _ in range(num_folds)]

    # for every group in datapoint_groups take a particular group and put each value from the specific group into the
    # specified folds
    for i, group in enumerate(datapoint_groups):
        for j, row in enumerate(group):
            fold_index = (j + i) % num_folds
            folds[fold_index].append(row)

    # make all folds np arrays
    folds = [np.array(fold) for fold in folds]
    data_folds = []
    label_folds = []

    # Create fold variables for labels and data
    for fold in folds:
        data_folds.append(fold[:, :-1])
        label_folds.append(fold[:, -1])

    return data_folds, label_folds


def get_folds_classification(data, labels, num_folds):

    # the get_folds function is meant to split the data up into a specified number of folds. this function takes in a
    # Dataset object as well as a specified number of folds. it then returns a list of all the data folds and label
    # folds

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
