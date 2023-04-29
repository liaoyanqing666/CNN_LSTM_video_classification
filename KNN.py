import pandas as pd
import numpy as np
from collections import Counter

# 欧氏距离
def euclidean_distance(x1, x2):
    distance = 0.0
    for i in range(len(x1)):
        distance += (x1[i] - x2[i]) ** 2
    return np.sqrt(distance)

# 曼哈顿距离
def manhattan_distance(x1, x2):
    distance = 0.0
    for i in range(len(x1)):
        distance += abs(x1[i] - x2[i])
    return distance

# 求与所有邻居之间的距离
def get_neighbors(x_train, y_train, x_test, k=1):
    distances = []
    for i in range(len(x_train)):
        dist = euclidean_distance(x_train[i], x_test) # 欧几里得距离
        # dist = manhattan_distance(x_train[i], x_test) # 曼哈顿距离
        distances.append((x_train[i], y_train[i], dist))
        distances.sort(key=lambda x: x[2])
    neighbors = []
    for i in range(k):
        neighbors.append((distances[i][0], distances[i][1]))
    class_counter = Counter([neighbor[1] for neighbor in neighbors])
    # print(class_counter.most_common(1)[0][1])
    return class_counter.most_common(1)[0][0] # 返回出现次数最多的标签

# 加载测试集和训练集
train_data = pd.read_csv('train//features.csv', header=None)
x_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
test_data = pd.read_csv('test//features.csv', header=None)
x_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

max_test = 400 # 最大测试数据
show = 50
print("欧几里得距离")
# print("曼哈顿距离")
accuracies = []
for k in range(1, 16, 2): # 计算k为单数
    y_pred = []
    for i in range(min(len(x_test), max_test)):
        if (i + 1) % show == 0:
            print(i + 1)
        result = get_neighbors(x_train, y_train, x_test[i], k=k)
        y_pred.append(result)
    y_test = y_test[:min(len(x_test), max_test)]
    accuracy = sum(y_pred == y_test) / len(y_test)
    print("k=", k, "Accuracy:", accuracy)
    accuracies.append((k, accuracy))
    pd.DataFrame([[k,accuracy]]).to_csv('euclidean_knn.csv', index=False, header=False, mode='a')
print("Accuracies:", accuracies)
