import torch
import torch.nn as nn
from torch.nn import functional as F

class SimpleCNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10):
        super(SimpleCNN, self).__init__()

        # 定义卷积层1：输入1通道，输出32通道，卷积核大小3x3
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)

        # 定义卷积层2：输入32通道，输出64通道
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)

        # 定义全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 输入大小 = 特征图大小 * 通道数
        self.fc2 = nn.Linear(128, num_classes)  # 10 个类别

    def forward(self, x):
        # 第一层卷积 + ReLU
        x = F.relu(self.conv1(x))
        # 最大池化
        x = F.max_pool2d(x, 2)

        # 第二层卷积 + ReLU
        x = F.relu(self.conv2(x))
        # 最大池化
        x = F.max_pool2d(x, 2)

        # 展平操作
        x = x.view(-1, 64 * 7 * 7)

        # 全连接层 + ReLU
        x = F.relu(self.fc1(x))
        # 全连接层输出
        x = self.fc2(x)
        return x
