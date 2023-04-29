import torch
from torch import nn, device
from dataset import *

# LSTM for video classification
class LSTM(nn.Module):
    def __init__(self, input_size=1000, hidden_size=300, num_layers=2, num_classes=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cnn_fc = nn.Linear(512 * 7 * 7, input_size) # cnn模型的全链接层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        x = self.cnn_fc(x)
        x = self.sigmoid(x)
        out_lstm, _ = self.lstm(x, (h0, c0))
        out_fc = self.fc(out_lstm[:, -1, :]) # 只取最后一次的输出
        out_lstm = out_lstm[:, -1, :]
        out_fc = self.sigmoid(out_fc)
        return out_fc, out_lstm # 返回out_lstm，便于调用训练好的LSTM可以直接返回全连接前视频的特征向量

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT) # 使用预训练好的模型进行迁移学习
        self.model = self.model.features # 只保留特征提取部分，大小512*7*7
        print(self.model)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

    def forward(self, x):
        b, f, c, h, w = x.size() # batchsize, frames, colors, height, width
        x = x.view(b * f, c, h, w)
        x = self.model(x)
        x = x.view(b, f, -1)
        return x


def accuracy(outputs, labels):
    right = 0
    outputs[outputs > 0.5] = 1
    outputs[outputs <= 0.5] = 0
    for i in range(len(labels)):
        if outputs[i] == labels[i]:
            right +=1
    return right


if __name__ == '__main__':
    cnn = CNN()
    lstm = LSTM(1000,200,3)
    try_dataset = dataset(max_num=1, video_frames=40)
    try_loader = DataLoader(try_dataset, batch_size=2, shuffle=True)
    for frame, label in try_loader:
        output_cnn = cnn(frame)
        print(output_cnn)
        output_lstm, _ = lstm(output_cnn)
        output_lstm = output_lstm.view(-1)
        print(output_lstm)
        print(label)
        print(accuracy(output_lstm, label))