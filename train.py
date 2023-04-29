import torch

from model import *
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def accuracy(outputs, labels):
    right = 0
    outputs[outputs > 0.5] = 1
    outputs[outputs <= 0.5] = 0
    for i in range(len(labels)):
        if outputs[i] == labels[i]:
            right +=1
    return right

# 初始化tensorboard
writer = SummaryWriter()


batch_size = 1
learning_rate = 0.01
epochs = 20
video_frame = 50
train_num_pair = 800 # 训练视频对数
test_num_pair = 200 # 测试视频对数
test_batch = 200 # 训练多少batch后测试一次
print_batch = 10 # 训练多少batch后输出/写入一次
shuffle_test = True
input_size = 1000
hidden_size = 400
num_layers = 4


train_set = dataset(train=True, video_frames=video_frame, max_num=train_num_pair)
test_set = dataset(train=False, video_frames=video_frame, max_num=test_num_pair, disorder=shuffle_test)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

cnn = CNN().to(device)
lstm = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
loss = nn.BCELoss().to(device)
optim = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

total_train_step = 0 # 便于画损失图
for i in range(epochs):
    print("-----第{}轮训练开始-----".format(i + 1))
    for now_batch, (frame, label) in enumerate(train_loader):
        frame = frame.to(device)
        label = label.to(device)
        output_cnn = cnn(frame)
        output_lstm, _ =lstm(output_cnn)
        output_lstm = output_lstm.view(-1)
        output_lstm = output_lstm.to(torch.float32)
        label = label.to(torch.float32)

        now_loss = loss(output_lstm, label)
        optim.zero_grad()
        now_loss.backward()
        optim.step()

        total_train_step += 1
        if now_batch % print_batch == 0:
            print("第{}轮训练第{}次迭代，训练损失函数{}".format(i + 1, now_batch + 1, now_loss))
            writer.add_scalar("train_loss (per {} iterations)".format(print_batch), now_loss, total_train_step)
        # if total_train_step % 50 == 0:
        #     torch.save(lstm.state_dict(), 'lstm_model.pth')

        if (now_batch + 1) % test_batch == 0:
            total_test_loss = 0.0
            total_right = 0
            total = 0
            with torch.no_grad():
                for now_batch, (frame, label) in enumerate(test_loader):
                    frame = frame.to(device)
                    label = label.to(device)
                    output_cnn = cnn(frame)
                    output_lstm, _ =lstm(output_cnn)
                    output_lstm = output_lstm.view(-1)
                    output_lstm = output_lstm.to(torch.float32)
                    label = label.to(torch.float32)

                    # 求总损失函数
                    now_loss = loss(output_lstm, label)
                    total_test_loss += now_loss
                    # 求正确个数
                    right = accuracy(output_lstm, label)
                    total += len(label)
                    total_right += right
            print("第{}轮训练，测试集总损失{}".format(i + 1, total_test_loss))
            print("第{}轮训练，测试集总正确率{}".format(i + 1, total_right / total))
            writer.add_scalar("test_loss", total_test_loss, total_train_step)
            writer.add_scalar("test_accuracy", total_right / total, total_train_step)
            torch.save(lstm.state_dict(), 'total_step{}_lstm_model.pth'.format(total_train_step)) # 便于找到过拟合前最合适的模型参数