import numpy as np
import matplotlib.pyplot as plt
from CreatePoint import X,labels
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_std = scaler.fit_transform(X)
cov_matrix = np.cov(X_std.T)
# print("Covariance Matrix:\n",cov_matrix)
eigen_values,eigen_vectors = np.linalg.eig(cov_matrix)
print("Eigen Values:\n",eigen_values)
print("Eigen Vectors:\n",eigen_vectors)

eigen_values_sorted_indices = np.argsort(eigen_values)[::-1] #按照从大到小排序
sorted_eigen_values = eigen_values[eigen_values_sorted_indices]
sorted_eigen_vectors = eigen_vectors[:,eigen_values_sorted_indices]

sorted_eigen_vectors[:, 0] = sorted_eigen_vectors[:, 0] * sorted_eigen_values[0]
sorted_eigen_vectors[:, 1] = sorted_eigen_vectors[:, 1] * sorted_eigen_values[1]
sorted_eigen_vectors[:, 2] = sorted_eigen_vectors[:, 2] * sorted_eigen_values[2]

W = sorted_eigen_vectors[:, 0:2]
# print("W:\n",W)
X_pca = X_std.dot(W)


if __name__ =="__main__":
    ax = plt.figure().add_subplot(111)
    scatter = plt.scatter(X_pca[:,0],X_pca[:,1],c=labels,cmap='viridis',s=20)
    legend = plt.legend(*scatter.legend_elements(),loc='upper right')
    plt.savefig('PCA.png')
    plt.show()