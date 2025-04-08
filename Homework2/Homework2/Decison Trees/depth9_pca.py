import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from PCA import X_pca,labels

X_train, X_test, y_train, y_test = train_test_split(X_pca, labels, test_size=0.2, random_state=0)

depth = 9
DT = DecisionTreeClassifier(max_depth=depth, random_state=42)
DT.fit(X_train, y_train)
y_pred_train = DT.predict(X_train)
y_pred_test = DT.predict(X_test)

acc_train = DT.score(X_train, y_train)
acc_test = DT.score(X_test, y_test)
print(f'Accuracy train on depth 9: {acc_train*100}%')
print(f'Accuracy test on depth 9: {acc_test*100}%')
# train:99.5625%;test:95.5%

# 训练集
train_class0_correct = (y_train == 0) & (y_pred_train == y_train)  # 类别 0 预测正确
train_class0_incorrect = (y_train == 0) & (y_pred_train != y_train)  # 类别 0 预测错误
train_class1_correct = (y_train == 1) & (y_pred_train == y_train)  # 类别 1 预测正确
train_class1_incorrect = (y_train == 1) & (y_pred_train != y_train)  # 类别 1 预测错误

# 测试集
test_class0_correct = (y_test == 0) & (y_pred_test == y_test)  # 类别 0 预测正确
test_class0_incorrect = (y_test == 0) & (y_pred_test != y_test)  # 类别 0 预测错误
test_class1_correct = (y_test == 1) & (y_pred_test == y_test)  # 类别 1 预测正确
test_class1_incorrect = (y_test == 1) & (y_pred_test != y_test)  # 类别 1 预测错误

# 创建网格用于绘制分类边界
h = 0.02  # 网格步长
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# 预测网格点的分类
Z = DT.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 创建子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 绘制训练集
ax1.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)  # 分类边界


ax1.scatter(X_train[train_class0_correct, 0], X_train[train_class0_correct, 1], c='g', marker='o',label='Class 0 Correct',s=20)
ax1.scatter(X_train[train_class0_incorrect, 0], X_train[train_class0_incorrect, 1],  c='red', label='Class 0 Incorrect', s=20)
ax1.scatter(X_train[train_class1_correct, 0], X_train[train_class1_correct, 1], c='blue', label='Class 1 Correct', s=20)
ax1.scatter(X_train[train_class1_incorrect, 0], X_train[train_class1_incorrect, 1], c='orange', label='Class 1 Incorrect', s=20)
ax1.set_title(f'Training Set \nAccuracy: {acc_train:.2f}')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.legend()

ax2.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
ax2.scatter(X_test[test_class0_correct, 0], X_test[test_class0_correct, 1],
            c='green', label='Class 0 Correct', s=20)  # 类别 0 正确 - 绿色
ax2.scatter(X_test[test_class0_incorrect, 0], X_test[test_class0_incorrect, 1],
            c='red', label='Class 0 Incorrect', s=20)  # 类别 0 错误 - 红色
ax2.scatter(X_test[test_class1_correct, 0], X_test[test_class1_correct, 1],
            c='blue', label='Class 1 Correct', s=20)  # 类别 1 正确 - 蓝色
ax2.scatter(X_test[test_class1_incorrect, 0], X_test[test_class1_incorrect, 1],
            c='orange', label='Class 1 Incorrect', s=20)  # 类别 1 错误 - 橙色
ax2.set_title(f'Test Set \nAccuracy: {acc_test:.2f} ')
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.legend()

plt.tight_layout()
plt.savefig('Depth9_pca.png')
plt.show()