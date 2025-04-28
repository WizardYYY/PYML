# -*- coding: gbk -*-
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载函数
def get_dataloaders(batch_size):
    train_trans = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_trans)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_trans)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

# get_dataloaders(32)


# CNN 模型
class CNN(nn.Module):
    def __init__(self,dropout_rate=0.35,out1=32,out2=64,lufunction=nn.ReLU,outfc=128):
        super(CNN, self).__init__()
        # TODO
        #juan ji ceng
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out1, out_channels=out2, kernel_size=3, padding=1)
        # chi hua ceng
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #quan lian jie ceng
        self.fc1 = nn.Linear(out2*7*7,outfc)
        self.fc2 = nn.Linear(outfc,10)
        self.relu = lufunction()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # TODO
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def count_parameters(model):
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    return total


model_test = CNN()
total = count_parameters(model_test)

##################################################
# 请勿修改此单元格中的代码
##################################################

# 评估指标累加器
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 计算准确率
def accuracy(y_hat, y_true):
    y_pred = y_hat.argmax(dim=1)
    return (y_pred == y_true).float().mean().item()



# 训练函数
def train_epoch(net, train_iter, loss_fn, optimizer):
    net.train()
    loop = tqdm(train_iter, desc='Train')
    device = next(net.parameters()).device
    metrics = Accumulator(3)
    for X, y in loop:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = net(X)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        metrics.add(loss.item() * len(y), accuracy(y_hat, y) * len(y), len(y))
    return metrics[0] / metrics[2], metrics[1] / metrics[2]

# 评估函数
@torch.no_grad()
def eval_model(net, test_iter, loss_fn):
    net.eval()
    device = next(net.parameters()).device
    metrics = Accumulator(3)
    for X, y in test_iter:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        loss = loss_fn(y_hat, y)
        metrics.add(loss.item() * len(y), accuracy(y_hat, y) * len(y), len(y))
    return metrics[0] / metrics[2], metrics[1] / metrics[2]


# # optuna调参示例
# import optuna
# import math
#
# # # 目标函数
# # def objective(trial):
# #     x = trial.suggest_float("x", 0, 2 * math.pi)
# #     y = trial.suggest_categorical("y", [1, 3, 5, 7, 9])
#
# #     value = math.sin(x) + math.log(y + 1)
# #     return value
#
# # # 调优过程
# # study = optuna.create_study(direction="maximize")
# # study.optimize(objective, n_trials=10)
#
# # # 输出结果
# # print("Best value:", study.best_value)
# # print("Best parameters:", study.best_params)

# Optuna 超参数优化
def objective(trial):
    # TODO
    epochs = range(1, 5)
    # (自行设置搜索参数与搜索范围)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [ 16, 32, 64, 128])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    out1 = trial.suggest_categorical("out1", [ 16, 32, 64, 128])
    out2 = trial.suggest_categorical("out2", [ 32, 64, 128])
    outfc = trial.suggest_categorical("outfc", [32, 64, 128, 256])
    lufunction = trial.suggest_categorical("lufunction", [nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.Tanh, nn.Sigmoid])

    trainloader, testloader = get_dataloaders(batch_size)
    model = CNN(dropout_rate=dropout_rate,out1=out1,out2=out2,outfc=outfc,lufunction=lufunction).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    patience = 3
    best_acc = float('inf')
    count = 0
    best_test_acc = 0
    for epoch in tqdm(epochs):
        train_epoch(model, trainloader, loss_fn, optimizer)
        _, test_acc = eval_model(model, testloader, loss_fn)
        # 加入早停机制
        if test_acc > best_test_acc:
            best_test_acc = max(best_test_acc, test_acc)
            count = 0
        else:
            count += 1
        if count >= patience:
            break

    return best_test_acc

import os
def train_best_model():
    # TODO
    if os.path.exists("best_model.pth"):
        # 可以从 optuna_results.csv 中加载最优参数
        if os.path.exists("optuna_results.csv"):
            df = pd.read_csv("optuna_results.csv")
            best_row = df.loc[df["test_acc"].idxmax()]  # 找到 test_acc 最大的一行
            print("最佳参数来自 optuna_results.csv:")
            print(best_row)
            activation_map = {
                "ReLU": nn.ReLU,
                "LeakyReLU": nn.LeakyReLU,
                "PReLU": nn.PReLU,
                "Tanh": nn.Tanh,
                "Sigmoid": nn.Sigmoid,
                "ELU": nn.ELU,
            }

            # 从字符串中解析出函数名，比如从 "<class 'torch.nn.modules.activation.PReLU'>" 变成 "PReLU"
            raw_func_str = str(best_row["lufunction"])
            try:
                act_name = raw_func_str.split("activation.")[-1].split("'")[0]
            except Exception as e:
                print("激活函数字符串解析失败，使用默认 ReLU。原始值：", raw_func_str)
                act_name = "ReLU"

            lufunc_obj = activation_map.get(act_name, nn.ReLU)  # 映射到实际函数对象

            params = {
                "batch_size": int(best_row["batch_size"]),
                "dropout_rate": best_row["dropout_rate"],
                "out1": int(best_row["out1"]),
                "out2": int(best_row["out2"]),
                "outfc": int(best_row["outfc"]),
                "lufunction": lufunc_obj,
                "lr": float(best_row["lr"]),
            }
            model = CNN(
                dropout_rate=params["dropout_rate"],
                out1=params["out1"],
                out2=params["out2"],
                lufunction=params["lufunction"],
                outfc=params["outfc"]
            ).to(device)
        else:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=20)

        model.load_state_dict(torch.load("best_model.pth"))
        print("Best model loaded")
        trainloader, testloader = get_dataloaders(params["batch_size"])
        loss_fn = nn.CrossEntropyLoss()
        _, test_acc = eval_model(model, testloader, loss_fn)
        print(f"Best test accuracy (from saved model): {test_acc:.4f}")
        return [], [], [], []
    else:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)
        # 保存结果到 CSV 文件
        results_df = pd.DataFrame([{**trial.params, "test_acc": trial.value} for trial in study.trials])
        results_df.to_csv("optuna_results.csv", index=False)

        # 输出最佳参数和结果
        best_trial = study.best_trial
        print(f"Best parameters: {best_trial.params}")
        print(f"Best test accuracy: {best_trial.value}")
        params = best_trial.params
        model = CNN(
            dropout_rate=params["dropout_rate"],
            out1=params["out1"],
            out2=params["out2"],
            lufunction=params["lufunction"],
            outfc=params["outfc"]
        ).to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])

        best_test_loss = float("inf")
        best_model_state = None
        early_stop_count = 0
        trainloader, testloader = get_dataloaders(params["batch_size"])
        epochs = range(1, 20+ 1)

        train_losses, test_losses, train_accs, test_accs = [], [], [], []
        for epoch in epochs:
            start_time = time.time()
            train_loss, train_acc = train_epoch(model, trainloader, loss_fn, optimizer)
            test_loss, test_acc = eval_model(model, testloader, loss_fn)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            end_time = time.time()
            train_time = end_time - start_time
            print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, Test Loss {test_loss:.4f}, Test Acc {test_acc:.4f}, Time {train_time:.4f}s")
            if test_loss<best_test_loss:
                best_test_loss = test_loss
                best_model_state = model.state_dict()
                early_stop_count = 0
                torch.save(best_model_state, "best_model.pth")
            else:
                early_stop_count += 1
            if early_stop_count >= 5:
                print("Early stopping")
                break
            print("Best test accuaracy: ", max(test_accs))
        return train_losses, test_losses, train_accs, test_accs

# 绘制学习曲线
def plot_learning_curves(train_losses, test_losses, train_accs, test_accs):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(train_losses, label='Train Loss')
    axs[0].plot(test_losses, label='Test Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[1].plot(train_accs, label='Train Accuracy')
    axs[1].plot(test_accs, label='Test Accuracy')
    axs[1].axhline(y=0.9, color='b', linestyle='--')  # 添加 y=0.9 的参考线
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    plt.savefig("learning_curves.png")
    plt.show()

def main():
    # 训练最终模型
    train_losses, test_losses, train_accs, test_accs = train_best_model()
    plot_learning_curves(train_losses, test_losses, train_accs, test_accs)

if __name__ == '__main__':
    main()