import torch.nn as nn
import torch

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers = 1):
        super(SimpleRNN, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # 定义 RNN 层
        # batch_first=True表示输入数据的维度为[batch_size, seq_len, input_szie]
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        """
            多种单层的 RNN 层
            单向、单层rnn, 1个时间步
                single_rnn = nn.RNN(input_size=4, hidden_size=3, num_layers=1, batch_first=True)
                input = torch.randn(1, 1, 4)    # 输入数据维度为[batch_size, time_steps_num, input_size], time_steps_num 实际上就是 input sequence length
                output, h_n = single_rnn(input) # output维度为[batch_size, time_steps_num, hidden_size=3]，h_n维度为[num_layers=1, batch_size, hidden_size=3]

            单向、单层rnn, 2个时间步
                single_rnn = nn.RNN(input_size=4, hidden_size=3, num_layers=1, batch_first=True)
                input = torch.randn(1, 2, 4) # 输入数据维度为[batch_size, time_steps_num, input_size]
                output, h_n = single_rnn(input) # output维度为[batch_size, time_steps_num, hidden_size=3]，h_n维度为[num_layers=1, batch_size, hidden_size=3]

            双向、单层rnn
                bi_rnn = nn.RNN(input_size=4, hidden_size=3, num_layers=1, batch_first=True, bidirectional=True)
                bi_output, bi_h_n = bi_rnn(input)
        """

        # 定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_prev = None):
        if h_prev == None:
            h_prev = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device = x.device, requires_grad=True) # 隐状态的形状为[层数，batch_size,hidden_size]
        # x: (batch_size, seq_len, input_size)
        out, hidden = self.rnn(x)  # out: (batch_size, seq_len, hidden_size)

        # 取序列最后一个时间步的输出作为模型的输出
        out = out[:, -1, :]  # (batch_size, hidden_size)

        out = self.fc(out)  # 全连接层
        return out, hidden
