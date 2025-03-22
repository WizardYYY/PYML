import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m

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

def gaussian_weighted_knn(x,y,x0,k,miu):
    y_pred = []
    for i in x0:
        distance = abs(x-i)
        index = np.argsort(distance)[:k]
        nearest_k = distance[index]
        nearest_k_y = y[index]
        weight = np.exp(-nearest_k**2/(2*miu**2))
        y_pred_value = np.sum(nearest_k_y*weight)/np.sum(weight)
        y_pred.append(y_pred_value)
    return np.array(y_pred)

x_line = np.linspace(0, 10, 1000)
y_pred = gaussian_weighted_knn(x_train,y_train,x_line,5,1)


# 绘制分段拟合结果
plt.figure(figsize=(10, 6))
plt.scatter(x_test, y_test, color='blue', marker='o', label='训练数据')

# 绘制曲线
plt.plot(x_line, y_pred, color='red', label = 'fitted_line')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('KNN_fitted')
plt.grid(True)
plt.legend()
plt.savefig('Other_method_test.png')
plt.show()

y_pred1 = gaussian_weighted_knn(x_train,y_train,x_train,5,1)
y_pred2 = gaussian_weighted_knn(x_train,y_train,x_test,5,1)
n = len(x_train)
loss1 = (1/n)*np.sum((y_pred1-y_train)**2)
loss2 = (1/n)*np.sum((y_pred2-y_test)**2)
print(f'{loss1},{loss2}')
