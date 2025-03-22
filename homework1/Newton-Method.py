import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 32行之前都是数据提取
path = 'Data4Regression.xlsx'
df1 = pd.read_excel(path, sheet_name=0,header=0)
# print(df1.head())
x_train = df1.iloc[0:,0].tolist()
y_train = df1.iloc[0:,1].tolist()
# print(len(x_train), len(y_train))

# #检测一下有没有正确读取
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

def loss(theta0,theta1,x,y):
    n = len(x)
    return (1/n)*np.sum((theta0+theta1*x-y)**2)

def df1(theta0,theta1,x,y):
    n = len(x)
    return (2/n)*np.sum(theta0+theta1*x-y)

def ddf1(theta0,theta1):
    return 2

def df2(theta0,theta1,x,y):
    n = len(x)
    return (2/n)*np.sum((theta0+theta1*x-y)*x)

def ddf2(theta0,theta1,x,y):
    n = len(x)
    return (2/n)*np.sum(x**2)

def NewtonMethod(x,y,tol,max_iter):
    theta0, theta1 = 0, 0
    loss_history = []
    for i in range(max_iter):
        theta0_new = theta0 - df1(theta0,theta1,x,y)/ddf1(theta0,theta1)
        theta1_new = theta1 - df2(theta0,theta1,x,y)/ddf2(theta0,theta1,x,y)
        loss_history.append(loss(theta0,theta1,x,y))
        if abs(theta0_new - theta0) < tol or abs(theta1_new - theta1) < tol:
            break
        theta0 = theta0_new
        theta1 = theta1_new
    return theta0, theta1, loss_history

theta0, theta1, loss_history = NewtonMethod(x_train,y_train,1e-6,100)
# x_line = np.linspace(0,10,10)
# y_pred = theta0 + theta1*x_line
# plt.scatter(x_train, y_train, color='blue', marker='o',label='datapoint')
# plt.plot(x_line, y_pred, color='red', label='fitted line')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.savefig('NewtonMethod_train.png')
# plt.show()
#
# plt.plot(loss_history,color='green',label='loss')
# plt.xlabel('iterations')
# plt.ylabel('loss')
# plt.legend()
# plt.savefig('NewtonMethod_loss.png')

# x_line = np.linspace(0,10,10)
# y_pred = theta0 + theta1*x_line
# plt.scatter(x_test, y_test, color='blue', marker='o',label='test_point')
# plt.plot(x_line, y_pred, color='red', label='fitted line')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.savefig('NewtonMethod_test.png')
# plt.show()

loss_test = loss(theta0,theta1,x_test,y_test)
print(theta0,theta1,loss_history[66],loss_test)
