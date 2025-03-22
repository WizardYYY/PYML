import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

array = np.ones(len(x_train))
X = np.hstack((array.reshape(-1,1), x_train.reshape(-1,1)))
Y = y_train.reshape(-1,1)
theta = np.array(2).T
Xt = X.T
XTX = np.dot(Xt, X)
XTY = np.dot(Xt, Y)
X0 = np.linalg.inv(XTX)
theta = np.dot(X0,XTY)
print(theta)

theta = theta.reshape(-1,2)
print(theta)
theta0 = theta[0,0]
theta1 = theta[0,1]

# x_line = np.linspace(0,10,10)
# y_pred = theta0 + theta1*x_line
# plt.scatter(x_train, y_train, color='blue', marker='o',label='datapoint')
# plt.plot(x_line, y_pred, color='red', label='fitted line')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.savefig('LeastSquare_train.png')
# plt.show()

# x_line = np.linspace(0,10,10)
# y_pred = theta0 + theta1*x_line
# plt.scatter(x_test, y_test, color='blue', marker='o',label='test_point')
# plt.plot(x_line, y_pred, color='red', label='fitted line')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.savefig('LeastSquare_test.png')
# plt.show()

loss_train = loss(theta0,theta1,x_train,y_train)
loss_test = loss(theta0,theta1,x_test,y_test)
print(theta0,theta1,loss_train,loss_test)