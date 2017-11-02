import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def LDA(training_data, training_labels, testing_data, testing_labels, number_of_samples):
    class_matrices = list()
    class_means = list()
    x = 0
    for i in range(0, 40):
        class_matrices.append(training_data[x:x+number_of_samples])
        class_means.append(np.mean(class_matrices[i], axis=0))
        x += number_of_samples

    total_mean = np.mean(training_data, axis=0)
    between_class_scatter_matrix = np.zeros((10304, 10304))
    print("calculating between class scatter matrix ...")
    for i in range(0, 40):
        delta_mean = class_means[i] - total_mean
        between_class_scatter_matrix += number_of_samples * delta_mean * delta_mean.T
    print("calculated between class scatter matrix ...")
    print("Data is now being Centered ...")
    for i in range(0, 40):
        class_matrices[i] = class_matrices[i] - class_means[i]
    print("Data is now Centered")
    print("calculating class scatter matrix ...")
    class_scatter_matrices = list()
    for i in range(0, 40):
        class_scatter_matrices.append(class_matrices[i].T * class_matrices[i])
    print("calculated class scatter matrix")
    print("calculating within class scatter matrix ...")
    within_class_scatter_matrix = np.zeros((10304, 10304))
    for i in range(0, 40):
        within_class_scatter_matrix += class_scatter_matrices[i]
    print("calculated within class scatter matrix")
    print("calculating inverse within class scatter matrix ...")
    inverse_within_class_scatter_matrix = np.linalg.inv(within_class_scatter_matrix)
    print("calculated inverse within class scatter matrix")
    print("calculating eigen values/vectors ...")
    eigens = np.linalg.eigh(inverse_within_class_scatter_matrix * between_class_scatter_matrix)
    print("calculated eigen values/vectors")
    eigen_values = eigens[0]
    eigen_vectors = eigens[1]
    np.save('LDA_eigen_values', eigen_values)
    np.save('LDA_eigen_vectors', eigen_vectors)
    # eigen_values = np.matrix(np.load('LDA_eigen_values.npy'))
    # eigen_vectors = np.matrix(np.load('LDA_eigen_vectors.npy'))
    print("values\n----------------------------")
    print(eigen_values)
    print("vectors\n----------------------------")
    print(eigen_vectors)
    projection_matrix = eigen_vectors[:, eigen_vectors.shape[1]-39:eigen_vectors.shape[1]]
    projected_data = training_data * projection_matrix
    kneighboursClassifier = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    kneighboursClassifier.fit(projected_data, training_labels.T)
    print(kneighboursClassifier.score(testing_data * projection_matrix, testing_labels.T))

