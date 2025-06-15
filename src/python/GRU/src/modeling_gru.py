import torch
import torch.nn as nn

class CustomGRU_layer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomGRU_layer, self).__init__()
        # 初始化参数
        self.W_xz = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_hz = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.W_xr = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_hr = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))

        self.hb_z = nn.Parameter(torch.zeros(hidden_size))
        self.hb_r = nn.Parameter(torch.zeros(hidden_size))
        self.hb_h = nn.Parameter(torch.zeros(hidden_size))
        self.xb_z = nn.Parameter(torch.zeros(hidden_size))
        self.xb_r = nn.Parameter(torch.zeros(hidden_size))
        self.xb_h = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, h):
        # 前向传播
        # 更新门
        z = torch.sigmoid((torch.matmul(x, self.W_xz) + self.xb_z) + (torch.matmul(h, self.W_hz) + self.hb_z))

        # 重置门
        r = torch.sigmoid((torch.matmul(x, self.W_xr) + self.xb_r) + (torch.matmul(h, self.W_hr) + self.hb_r))

        # 候选隐藏状态
        h_tilda = torch.tanh((torch.matmul(x, self.W_xh) + self.xb_h) + r * (torch.matmul(h, self.W_hh) + self.hb_h))

        # 更新隐藏状态
        h = z * h + (1 - z) * h_tilda

        return h

class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, device = None):
        super(CustomGRU, self).__init__()
        # 输入特征的维度
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device if device is not None else "cpu"

        # 初始化自定义的GRU层
        self.gru = CustomGRU_layer(self.input_size, self.hidden_size)

    def forward(self, X, h0 = None):
        # x.shape = (seq_length, batch_size, input_size)
        # h0.shape = (1, batch_size, hidden_size)
        # output.shape = (seq_length, batch_size, hidden_size)

        # 获取批次大小
        batch_size = X.shape[1]
        # 获取序列长度
        self.seq_length = X.shape[0]

        # 如果没有提供初始隐藏状态，则初始化为零张量
        if h0 is None:
            prev_h = torch.zeros([batch_size, self.hidden_size]).to(self.device)
        else:
            prev_h = torch.unsqueeze(h0, 0).to(self.device)

        # 初始化输出张量
        output = torch.zeros([self.seq_length, batch_size, self.hidden_size]).to(self.device)

        # 循环处理序列中的每个时间步
        for index in range(self.seq_length):
            # 通过GRU层处理当前时间步的数据，并更新隐藏状态
            prev_h = self.gru(X[index], prev_h)
            # 将当前时间步的输出存储在输出张量中
            output[index] = prev_h

        # 返回最终的输出和隐藏状态
        return output, torch.unsqueeze(prev_h, 0)


class GRUModel(nn.Module):
    def __init__(self, input_size, num_class, hidden_size = 768, middle_hidden_size = 256, device = None):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = CustomGRU(input_size, hidden_size, device)
        # self.gru = nn.GRU(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, middle_hidden_size)
        self.fc2 = nn.Linear(middle_hidden_size, num_class)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        embed = self.embedding(inputs)
        embed = self.relu(embed)

        out, _ = self.gru(embed)
        out = self.relu(out)

        fc1 = self.fc1(out)
        fc1 = self.relu(fc1)
        fc2 = self.fc2(fc1)
        return fc2
