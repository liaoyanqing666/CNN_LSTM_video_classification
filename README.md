# CNN_LSTM_video_classification

> dataset download: https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset

## 项目介绍

本项目使用CNN2d-LSTM模型实现对暴力视频的鉴别任务

前向传递思想：使用CNN提取每一帧特征，之后将提取出的所有特征送入LSTM计算视频特征，接着对视频特征采用不同方式进行分类。

反向传递思想：CNN-LSTM模型中，LSTM每次输入都连接一个CNN，无法反向传递（或者过于困难），因此采用在ImageNet预训练好的CNN模型，从而只需迭代LSTM。


1. 截取每个视频长度至相同帧数（超参数可设置），提取帧特征并送入LSTM，开始训练LSTM。

2. 此时需在LSTM最终输出后连接一个全连接层，便于训练时反向传递。

3. 训练好后，将LSTM的输出保存至CSV文件，即视频的特征及标签。

4. 采用不同方法对CSV文件进行分类。

## English translation of project introduction

This project uses a CNN2d-LSTM model to perform the task of discriminating violent videos.

Forward propagation: The idea is to use a CNN to extract features for each frame, then feed all the extracted features into an LSTM to calculate the video features, and finally classify the video features using different methods.

Backward propagation: In a CNN-LSTM model, each LSTM input is connected to a CNN and cannot be backpropagated (or is too difficult to do so), so a pre-trained CNN model on ImageNet is used instead, and only the LSTM is iterated.

The length of each video is truncated to the same number of frames (can be set as a hyperparameter), and the frame features are extracted and fed into the LSTM to start training.

At this point, a fully connected layer should be connected to the LSTM's final output to facilitate backpropagation during training.

Once trained, the LSTM's output is saved to a CSV file, including the video features and labels.

Different methods are used to classify the CSV file.
