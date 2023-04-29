import random
from torch.utils.data import Dataset, DataLoader
import os, cv2, torch
import torchvision
import numpy as np

class dataset(Dataset):
    # 正样本数需要等于负样本数
    def __init__(self, train=True, max_num=100000000, video_width=224, video_height=224, video_frames=60, disorder=False):
        # max_num代表最多多少对正负样本
        # video_frames代表视频总帧数
        # disorder是否打乱的视频的顺序，避免max_num较小时读入多个相近视频

        super(dataset, self).__init__()
        if train:
            self.path = 'train/'
        else:
            self.path = 'test/'

        self.max_num = max_num
        self.video_width = video_width
        self.video_height = video_height
        self.video_frames = video_frames
        self.pos_list = os.listdir(self.path + 'violence') # 正训练样本文件名列表
        self.neg_list = os.listdir(self.path + 'nonviolence') # 负训练样本文件名列表
        if disorder:
            random.shuffle(self.pos_list)
            random.shuffle(self.neg_list)

    def __len__(self):
        return min(len(self.pos_list)+len(self.neg_list), 2 * self.max_num)

    def __getitem__(self, item): # item大于等于正样本数则返回负样本，小于则正样本

        if 2 * self.max_num < len(self.pos_list)+len(self.neg_list):
            if item >= self.max_num:
                item -= self.max_num
                path = self.path + 'nonviolence/' + self.neg_list[item]
                label = 0
            else:
                path = self.path + 'violence/' + self.pos_list[item]
                label = 1
        else:
            if item >= len(self.pos_list):
                item -= len(self.pos_list)
                path = self.path + 'nonviolence/' + self.neg_list[item]
                label = 0.0
            else:
                path = self.path + 'violence/' + self.pos_list[item]
                label = 1.0

        video = cv2.VideoCapture(path) # 读取视频文件
        num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT) # 读取视频总帧数
        frame_list = np.floor(np.linspace(1, num_frames, self.video_frames))  # 在视频中选择video_frame帧
        # print(frame_list)

        i = 0.0  # 计数器
        frames = torch.zeros((1, 3, self.video_height, self.video_width), dtype=torch.float32)  # 创建一个 Tensor 存储视频帧
        while True:
            ret, frame_video = video.read() # 逐帧读取数据
            if not ret:
                break
            i += 1
            while i == frame_list[0]: # 当前读取到了第i帧，查看所需要的帧序列里是否有第i帧，有几个第i帧
                frame = cv2.resize(frame_video, (224, 224))
                frame = torch.from_numpy(frame.transpose((2, 0, 1))).unsqueeze(0).float() / 255.0  # 转成tensor并归一化
                frames = torch.cat((frames, frame), dim=0)
                frame_list = frame_list[1:]
                if len(frame_list) == 0:
                    break
            if len(frame_list) == 0:
                break

        frames = frames[1:]  # 删除第一个全 0 的 Tensor
        return frames, torch.tensor(label)


if __name__ == "__main__":
    try_dataset = dataset(train=False,video_frames=100, disorder=True)
    try_loader = DataLoader(try_dataset, batch_size=2, shuffle=True)
    for frame, label in try_loader:
        print('frame', frame.shape)
        print('label', label)

