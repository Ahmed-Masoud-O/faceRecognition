import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def PCA(training_data, alpha, training_labels, testing_data, testing_labels):
    print("------------\n Mean \n------------")
    meanVector = np.mean(training_data, 0)
    print(meanVector)

    centeredMatrix = training_data - meanVector
    print("----------------\n Centered Matrix \n----------------")
    print(centeredMatrix)

    print("----------------- \nCovariance Matrix \n-----------------")
    covarianceMatrix = np.dot(centeredMatrix.T, centeredMatrix) * (1 / training_data.shape[0])
    print(covarianceMatrix)
    print(covarianceMatrix.shape)

    eigens = np.linalg.eigh(covarianceMatrix)
    eigenValues = eigens[0]
    eigenVector = eigens[1]
    # eigenValues = np.matrix(np.load('eigenValues.npy'))
    # eigenVector = np.matrix(np.load('eigenVectors.npy'))
    print("--------------- \n eigen values \n --------------")
    print(eigenValues)
    print(eigenValues.shape)
    print("--------------- \n eigen Vectors \n --------------")
    print(eigenVector)
    np.save('eigenValues', eigenValues)
    np.save('eigenVectors', eigenVector)
    eigen_sum = eigenValues.sum()
    my_sum = 0
    for x in range(0, eigenValues.shape[1]):
        my_sum += eigenValues[0, eigenValues.shape[1]-x-1]
        if my_sum/eigen_sum >= alpha:
            break
    print(x)
    projection_matrix = eigenVector[:, eigenValues.shape[1]-x:eigenValues.shape[1]]
    print(projection_matrix.shape)
    projected_data = training_data * projection_matrix
    kneighboursClassifier = KNeighborsClassifier(n_neighbors=1)
    kneighboursClassifier.fit(projected_data, training_labels.T)
    print(kneighboursClassifier.score(testing_data * projection_matrix, testing_labels.T))
