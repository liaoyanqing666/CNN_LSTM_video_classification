# 计算每个视频的特征向量并保存
from model import *
from dataset import *
import pandas as pd
torch.set_printoptions(precision=20)
pd.set_option('display.float_format', lambda x: f'{x:.15f}')


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    cnn = CNN().to(device)
    lstm = LSTM(input_size=1000, hidden_size=400, num_layers=4).to(device)
    lstm.load_state_dict(torch.load('total_step8400_lstm_model.pth', map_location=device))

    train_dataset = dataset(train=True, video_frames=50, disorder=False)
    test_dataset = dataset(train=False, video_frames=50, disorder=False)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, (frame, label) in enumerate(train_loader):
            frame = frame.to(device)
            label = label.view(1,1)
            label = label.to(device)
            output_cnn = cnn(frame)
            _, feature = lstm(output_cnn)
            feature = torch.cat([feature, label], dim=1)
            feature = feature.cpu()
            pd.DataFrame(feature.detach().numpy()).to_csv('train\\features.csv', index=False, header=False, mode='a', float_format='%.20f')
            print('train:', i+1)

    with torch.no_grad():
        for i, (frame, label) in enumerate(test_loader):
            frame = frame.to(device)
            label = label.view(1,1)
            label = label.to(device)
            output_cnn = cnn(frame)
            _, feature = lstm(output_cnn)
            feature = torch.cat([feature, label], dim=1)
            feature = feature.cpu()
            pd.DataFrame(feature.detach().numpy()).to_csv('test\\features.csv', index=False, header=False, mode='a', float_format='%.20f')
            print('test:', i+1)

