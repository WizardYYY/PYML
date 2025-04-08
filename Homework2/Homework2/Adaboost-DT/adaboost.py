import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from CreatePoint import X,labels

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=0)

train_errors = []
test_errors = []
n_estimators = 10
for i in range(1,n_estimators+1):
    adaclf = AdaBoostClassifier(
        estimator = DecisionTreeClassifier(max_depth=5, random_state=42),
        n_estimators=i,
        random_state=42,
    )
    adaclf.fit(X_train, y_train)
    train_error = 1-adaclf.score(X_train, y_train)
    test_error = 1-adaclf.score(X_test, y_test)
    train_errors.append(train_error)
    test_errors.append(test_error)
    if i == n_estimators:
        print(f"{adaclf.score(X_train, y_train)*100}%")
        print(f"{adaclf.score(X_test, y_test)*100}%")
        # train:99.8125%;test:96.25%
        y_pred_train = adaclf.predict(X_train)
        y_pred_test = adaclf.predict(X_test)
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

        fig = plt.figure()
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(X_train[train_class0_correct, 0], X_train[train_class0_correct, 1],
                    X_train[train_class0_correct, 2], c='g', marker='o', label='Class 0 Correct')
        ax1.scatter(X_train[train_class0_incorrect, 0], X_train[train_class0_incorrect, 1],
                    X_train[train_class0_incorrect, 2], c='red', label='Class 0 Incorrect', s=50)
        ax1.scatter(X_train[train_class1_correct, 0], X_train[train_class1_correct, 1],
                    X_train[train_class1_correct, 2], c='blue', label='Class 1 Correct', s=50)
        ax1.scatter(X_train[train_class1_incorrect, 0], X_train[train_class1_incorrect, 1],
                    X_train[train_class1_incorrect, 2], c='orange', label='Class 1 Incorrect', s=50)
        ax1.set_title('Training Set ')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(X_test[test_class0_correct, 0], X_test[test_class0_correct, 1], X_test[test_class0_correct, 2],
                    c='green', label='Class 0 Correct', s=50)  # 类别 0 正确 - 绿色
        ax2.scatter(X_test[test_class0_incorrect, 0], X_test[test_class0_incorrect, 1],
                    X_test[test_class0_incorrect, 2],
                    c='red', label='Class 0 Incorrect', s=50)  # 类别 0 错误 - 红色
        ax2.scatter(X_test[test_class1_correct, 0], X_test[test_class1_correct, 1], X_test[test_class1_correct, 2],
                    c='blue', label='Class 1 Correct', s=50)  # 类别 1 正确 - 蓝色
        ax2.scatter(X_test[test_class1_incorrect, 0], X_test[test_class1_incorrect, 1],
                    X_test[test_class1_incorrect, 2],
                    c='orange', label='Class 1 Incorrect', s=50)  # 类别 1 错误 - 橙色
        ax2.set_title('Test Set ')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()

        plt.tight_layout()
        plt.savefig('Accuracy Plot.png')
        plt.show()

        print("各分类器的权重：")
        for k,weight in enumerate(adaclf.estimator_weights_):
            print(f"第{k}个的权重: {weight}")

plt.plot(range(1,n_estimators+1), train_errors, label='train errors',marker='o',color='blue')
plt.plot(range(1,n_estimators+1), test_errors, label='test errors',marker='s',color='orange')
plt.xlabel('epochs')
plt.ylabel('error')
plt.title('AdaBoost_error_epoch')
plt.legend()
plt.savefig('AdaBoost_error_epoch.png')
plt.show()
