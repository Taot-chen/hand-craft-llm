import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from utils import MAX_LENGTH, SOS_TOKEN, EOS_TOKEN


# Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, device=None):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.device = "cpu" if device is None else device

        # 词嵌入层
        self.embedding = nn.Embedding(
            input_size, hidden_size, device=self.device
        )
        # GRU层
        self.gru = nn.GRU(
            hidden_size, hidden_size, device=self.device
        )

    def forward(self, input_tensor, hidden):
        embed = self.embedding(input_tensor).view(1, 1, -1) # (1, 1, hidden_size)
        output, hidden = self.gru(
            embed, hidden
        )  # (1, 1, hidden_size) 和 (1, 1, hidden_size)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(
            1, 1, self.hidden_size, device=self.device
        )  # 初始化隐藏状态 (1, 1, hidden_size)


# Decoder with attention
class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout=0.1, max_length=MAX_LENGTH, device=None):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.device = "cpu" if device is None else device

        # 词嵌入层
        self.embedding = nn.Embedding(
            output_size, hidden_size, device=self.device
        )
        # 计算注意力权重
        self.attn = nn.Linear(2 * hidden_size, max_length).to(self.device)
        # 合并嵌入向量和注意力加权值
        self.attn_combine = nn.Linear(2 * hidden_size, hidden_size).to(
            self.device
        )
        # GRU层
        self.gru = nn.GRU(hidden_size, hidden_size).to(self.device)
        # Dropout层
        self.dropout = nn.Dropout(dropout).to(self.device)
        # 输出层
        self.output = nn.Linear(hidden_size, output_size).to(self.device)

    def forward(self, input, hidden, encoder_outputs):
        embed = self.embedding(input).view(1, 1, -1)   # (1, 1, hidden_size)
        dropout = self.dropout(embed)   # (1, 1, hidden_size)

        # 计算注意力权重
        attn_weights = F.softmax(
            self.attn(torch.cat((dropout[0], hidden[0]), dim=1)), dim=1
        )  # (1, max_length)

        # 计算加权后的上下文向量
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0)
        )  # (1, 1, hidden_size)

        # 拼接嵌入向量和上下文向量
        output = torch.cat((embed[0], attn_applied[0]), 1)  # (1, 2 * hidden_size)
        output = self.attn_combine(output).unsqueeze(0)  # (1, 1, hidden_size)
        output = F.relu(output)  # (1, 1, hidden_size)

        output, hidden = self.gru(
            output, hidden
        )  # (1, 1, hidden_size) 和 (1, 1, hidden_size)
        output = self.output(output[0])  # (1, output_size)
        output = F.log_softmax(output, dim=1)  # (1, output_size)

        return output, hidden, attn_weights


class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder, device=None):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = "cpu" if device is None else device

    def forward(self, input_tensor, target_tensor, teacher_forcing_ratio=0.5):
        # 输入序列的长度
        input_length = input_tensor.size(0)
        # 目标序列的长度
        target_length = target_tensor.size(0)

        encoder_hidden = self.encoder.init_hidden()  
        encoder_outputs = torch.zeros(
            MAX_LENGTH, self.encoder.hidden_size, device=self.device
        )

        # 编码阶段
        for index in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[index], encoder_hidden
            )  # (1, 1, hidden_size) 和 (1, 1, hidden_size)
            encoder_outputs[index] = encoder_output[
                0, 0
            ]  # 取出每个时间步的输出 (MAX_LENGTH, hidden_size)

        # 初始化解码器输入（开始符号）和隐藏状态
        decoder_input = torch.tensor([[SOS_TOKEN]], device=self.device)  # (1, 1)
        decoder_hidden = encoder_hidden
        all_decoder_outputs = torch.zeros(
            target_length, self.decoder.output_size, device=self.device
        )   # (target_length, output_size)

        use_teacher_force = random.random() < teacher_forcing_ratio  # 是否使用教师强制

        # 解码阶段
        for index in range(target_length):
            # (1, output_size), (1, 1, hidden_size), (1, max_length)
            decoder_output, decoder_hidden, attn_weights = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # 存储每一步的输出 (target_length, output_size)
            all_decoder_outputs[index] = (
                decoder_output
            )

            # 获取最大概率的词索引
            topv, topi = decoder_output.topk(1)
            # 获取下一个时间步的输入 (1)
            decoder_input = topi.squeeze().detach()
            # 使用教师标签作为下一步的输入
            if use_teacher_force:
                decoder_input = target_tensor[index]

        # (target_length, output_size)
        return all_decoder_outputs
