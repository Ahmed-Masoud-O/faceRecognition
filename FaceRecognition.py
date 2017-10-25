import numpy as np
import os
from scipy.misc import imread
import PCA


def load(my_folder='orl_faces', train_count=5, test_count=5):
    training_data = []
    training_labels = []
    testing_data = []
    testing_labels = []
    for folder in os.listdir(my_folder):
        path = my_folder + '/' + folder
        if os.path.isdir(path):
            files = os.listdir(path)
            for i in range(0, train_count):
                training_data.append(imread(path + '/' + files[i]).flatten())
                training_labels.append(folder)
            for i in range(train_count, train_count + test_count):
                testing_data.append(imread(path + '/' + files[i]).flatten())
                testing_labels.append(folder)
    return np.asmatrix(training_data), np.asmatrix(testing_data), np.asmatrix(training_labels), np.asmatrix(testing_labels)

training_data, testing_data, training_labels, testing_labels = load()

# PCA

PCA.PCA(training_data, 0.95, training_labels, testing_data, testing_labels)



