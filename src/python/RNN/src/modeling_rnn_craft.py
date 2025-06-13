import torch
import torch.nn as nn


class RNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.bidirectional = False
        self.W_hh = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
        self.W_ih = nn.Parameter(torch.rand(self.input_size, self.hidden_size))
        self.b_ih = nn.Parameter(torch.zeros(self.hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(self.hidden_size))

    def forward(self, x_t, h_prev=None):
        """
            1: torch.matmul(x_t, self.W_ih)
                x_t包含多个时间步，形状为[batch_size, time_steps_num, input_size]   # input_size 实际上就是 input 的维度 input_dim
                W_ih形状为[input_size, hidden_size]
                torch.matmul(x_t, self.W_ih) 输出矩阵形状为[batch_size, time_steps_num, hidden_size]

            2: torch.matmul(h_prev, self.W_hh)
                h_prev 形状为[batch_size, time_steps_num, hidden_size]
                W_hh形状为[hidden_size, hidden_size]
                torch.matmul(h_prev, self.W_hh) 输出矩阵形状为[batch_size, time_steps_num, hidden_size]
        """
        output = torch.matmul(x_t, self.W_ih) + self.W_ih + torch.matmul(h_prev, self.W_hh) + self.b_hh
        output = torch.tanh(output)
        return output, output[:, -1, :].unsqueeze(0)


class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers = 1):
        super(CustomRNN, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # 定义 RNN 层
        self.rnn = RNNLayer(input_size = input_size, hidden_size = hidden_size, num_layers = 1, batch_first = True)

        # 定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, h_prev):
        if h_prev == None:
            h_prev = torch.zeros(self.num_layers, inputs.size(0), self.hidden_size, device = inputs.device, requires_grad=True) # 隐状态的形状为[层数，batch_size,hidden_size]
        out, h = self.rnn(inputs, h_prev)
        out = self.fc(out[:, -1, :])    # 取最后一个时间步的输出作为网络的输出
        return out, h
