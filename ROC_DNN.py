# 绘制ROC曲线
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from DNN import *
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc_curve(y_true, y_pred, label):
    # 将真实标签和预测标签转换为numpy数组
    y_true = np.array(y_true.detach())
    y_pred = np.array(y_pred.detach())

    # 计算ROC曲线上的点
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    # 计算AUC值
    auc_value = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.plot(fpr, tpr, label='{} ROC area = %0.6f'.format(label) % auc_value)
    plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve'.format(label))
    plt.legend(loc="lower right")


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    dnn_relu = torch.load('dnn_RELU.pth')
    dnn_relu = dnn_relu.to(device)
    dnn_sigmoid = torch.load('dnn_SIGMOID.pth')
    dnn_sigmoid = dnn_sigmoid.to(device)
    dnn_tanh = torch.load('dnn_TANH.pth')
    dnn_tanh = dnn_tanh.to(device)

    test_set = dataset(train=False)
    test_loader = DataLoader(test_set, batch_size=400, shuffle=False)  # 一次全部计算完

    for x_test, y_test in test_loader:
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        y_pre = dnn_relu(x_test)
        plot_roc_curve(y_test, y_pre, 'relu')

    for x_test, y_test in test_loader:
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        y_pre = dnn_tanh(x_test)
        plot_roc_curve(y_test, y_pre, 'tanh')

    for x_test, y_test in test_loader:
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        y_pre = dnn_sigmoid(x_test)
        plot_roc_curve(y_test, y_pre, 'sigmoid')

    plt.savefig('ROC curve', dpi=600)
    plt.show()