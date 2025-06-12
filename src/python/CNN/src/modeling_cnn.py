import torch
import torch.nn as nn
from torch.nn import functional as F

class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        # images input channle = 1
        # output channel = 6
        # conv kernel 5x5
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # affine operator for: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):
        # max_pooling window size: 2x2
        # 经过relu激活函数，再经过max_pooling池化
        input = self.conv1(input)
        input = F.relu(input)
        input = F.max_pool2d(input, (2, 2))

        # If the size is a square you can only specify a single number for pool window size
        input = self.conv2(input)
        input = F.relu(input)
        input = F.max_pool2d(input, 2)

        # 铺平
        input = input.view(-1, self.flat_feature(input))

        input = self.fc1(input)
        input = F.relu(input)

        input = self.fc2(input)
        input = F.relu(input)
        input = self.self(input)
        return input


    def flat_feature(self, input):
        sizes = input.size[1:]   # all dimensions except the batch dimension
        num_feat = 1
        for size in sizes:
            num_feat *= size
        return num_feat

