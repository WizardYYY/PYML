import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from CreatePoint import X,labels
from PCA import X_pca,labels


X_train, X_test, y_train, y_test = train_test_split(X_pca, labels, test_size=0.2, random_state=0)
# X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=0)

# 检验一下是不是导入成功
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# scatter = ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test, cmap='viridis', marker='o')
# legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
# ax.add_artist(legend1)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.title('Point test')
# plt.savefig('Testpoints.png')
# plt.show()

depths = range(1,15)
training_errors = []
test_errors = []

for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)

    training_errors.append(1-clf.score(X_train, y_train))
    test_errors.append(1-clf.score(X_test, y_test))

# 绘制层深与误差
plt.figure(figsize=(8,4))
plt.plot(depths, training_errors, label='Training Error', color='blue',marker='o')
plt.plot(depths, test_errors, label='Test Error', color='red',marker='s')
plt.xlabel('Decision Tree Depth')
plt.ylabel('Error')
plt.title('Error vs Decision Tree Depth')
plt.legend()
plt.savefig('Error_Depth_pca.png')
# plt.savefig('Error_Depth.png')
plt.show()