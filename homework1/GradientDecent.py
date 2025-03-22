import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

path = 'Data4Regression.xlsx'
df1 = pd.read_excel(path, sheet_name=0,header=0)
# print(df1.head())
x_train = df1.iloc[0:,0].tolist()
y_train = df1.iloc[0:,1].tolist()
# print(len(x), len(y))

# 检测一下有没有正确读取
# plt.plot(x_train,y_train,marker='o',linestyle='None',color='red',label='Data')
# plt.grid(True)
# plt.savefig('Dot_train.png')
# plt.show()

df2 = pd.read_excel(path, sheet_name=1,header=0)
# print(df2.head())
x_test = df2.iloc[0:,0].tolist()
y_test = df2.iloc[0:,1].tolist()
#print(len(x), len(y))

# plt.plot(x_test,y_test,marker='o',linestyle='None',color='red',label='Data')
# plt.grid(True)
# plt.savefig('Dot_test.png')
# plt.show()

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

def gradient_descent(x,y,learning_rate ,epochs):
    n = len(x)
    theta0,theta1 = 0,0
    loss_history = []
    for i in range(epochs):
        y_pred = theta0 + theta1*x
        loss = (1/n)*np.sum((y_pred - y)**2)
        loss_history.append(loss)

        dtheta0 = (2/n)*np.sum(y_pred - y)
        dtheta1 = (2/n)*np.sum((y_pred - y)*x)
        # dtheta2 = (2/n)*np.sum((y_pred - y)*(x**2))

        theta0 = theta0 - learning_rate*dtheta0
        theta1 = theta1 - learning_rate*dtheta1
        # theta2 = theta2 - learning_rate*dtheta2
    return theta0,theta1,loss_history

theta0,theta1,loss_history = gradient_descent(x_train,y_train,0.01,100)
x_line = np.linspace(0,10,10)
y_line = theta0 + theta1*x_line
# plt.scatter(x_train,y_train,color='blue',marker='o',label='DataPoint')
# plt.plot(x_line,y_line,color='red',label='fitted line')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.savefig('GradientDescent_Train.png')
# plt.show()
#
# plt.plot(loss_history,color = 'green')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.yscale('log')
# plt.savefig('GradientDescent_Train_Loss.png')

# # 测试测试集的情况
# plt.scatter(x_test,y_test,color='blue',marker='o',label='DataPoint')
# plt.plot(x_line,y_line,color='red',label='fitted line')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.savefig('GradientDescent_Test.png')
# plt.show()

loss_test = (1/len(x_test)) * np.sum((theta0+theta1*x_test - y_test)**2)
print(theta0,theta1,loss_history[99],loss_test)
