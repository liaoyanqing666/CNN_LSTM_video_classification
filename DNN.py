# 计算三种不同激活函数下激活层的区别
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


class DNN(nn.Module):
    def __init__(self, activation, input_size=400, output_size=1):
        super(DNN, self).__init__()

        if activation == 1:
            self.linear = nn.Sequential( # relu
                nn.Linear(input_size, 1000),
                nn.ReLU(),
                nn.Linear(1000, 300),
                nn.ReLU(),
                nn.Linear(300, output_size),
                nn.Sigmoid()
            ).to(device)
        elif activation == 2:
            self.linear = nn.Sequential( # sigmoid
                nn.Linear(input_size, 1000),
                nn.Sigmoid(),
                nn.Linear(1000, 300),
                nn.Sigmoid(),
                nn.Linear(300, output_size),
                nn.Sigmoid()
            ).to(device)
        elif activation == 3:
            self.linear = nn.Sequential( # tanh
                nn.Linear(input_size, 1000),
                nn.Tanh(),
                nn.Linear(1000, 300),
                nn.Tanh(),
                nn.Linear(300, output_size),
                nn.Sigmoid()
            ).to(device)

    def forward(self, x):
        predict = self.linear(x)
        return predict


class dataset(Dataset):
    def __init__(self, train=True):
        super(dataset, self).__init__()
        if train:
            train_data = pd.read_csv('train//features.csv', header=None)
            self.x = train_data.iloc[:, :-1].values
            self.y = train_data.iloc[:, -1].values
        else:
            test_data = pd.read_csv('test//features.csv', header=None)
            self.x = test_data.iloc[:, :-1].values
            self.y = test_data.iloc[:, -1].values

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return torch.tensor(self.x[item], dtype=torch.float32), (torch.tensor(self.y[item], dtype=torch.float32)).view(-1)


def accuracy(outputs, labels):
    right = 0
    outputs[outputs > 0.5] = 1
    outputs[outputs <= 0.5] = 0
    for i in range(len(labels)):
        if outputs[i] == labels[i]:
            right +=1
    return right


if __name__ == '__main__':
    # 总样本1600训练，400测试

    activation = 3 # 这里可以调整激活函数
    if activation == 1:
        active = 'RELU'
    elif activation == 2:
        active = 'SIGMOID'
    elif activation == 3:
        active = 'TANH'

    batch_size = 20
    learning_rate = 0.01
    if activation == 1:
        epoch = 10
    if activation == 2:
        epoch = 30
    if activation == 3:
        epoch = 6
    cal_acc_iter = 80 # 迭代多少次计算一次正确率

    # writer = SummaryWriter()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_set = dataset(train=True)
    test_set = dataset(train=False)
    train_loader = DataLoader(train_set, batch_size=batch_size ,shuffle=True)
    test_loader = DataLoader(test_set, batch_size=400, shuffle=False) # 一次全部计算完
    dnn = DNN(activation=activation).to(device)
    loss = nn.BCELoss().to(device)
    optim = torch.optim.SGD(dnn.parameters(), lr=learning_rate)

    for i in range(epoch):
        step = 0
        print('第{}轮训练'.format(i + 1))
        for x_train, y_train in train_loader:
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            y_pre = dnn(x_train)
            train_loss = loss(y_pre, y_train)
            optim.zero_grad()
            train_loss.backward()
            optim.step()
            step += 1
            # print('第{}轮第{}次迭代，训练损失函数:'.format(i+1, step), train_loss)
            # writer.add_scalar('train_loss_{}'.format(active), train_loss, i * 1600 + step * batch_size)

            if step % cal_acc_iter == 0:
                for x_test, y_test in test_loader:
                    x_test = x_test.to(device)
                    y_test = y_test.to(device)
                    y_pre = dnn(x_test)
                    # plot_roc_curve(y_test, y_pre)
                    acc = accuracy(y_pre, y_test)/len(y_test)
                    print('第{}轮第{}次迭代，测试正确率:'.format(i+1, step), acc)
                    # writer.add_scalar('test_accuracy_{}'.format(active), acc, i * 1600 + step * batch_size)
    torch.save(dnn, 'dnn_{}.pth'.format(active))




