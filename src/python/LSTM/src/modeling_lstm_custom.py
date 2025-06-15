import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam 
import lightning as L
from torch.utils.data import TensorDataset, DataLoader


class CraftLSTM(L.LightningModule):
    def __init__(self):
        super().__init__()
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)

        # 遗忘门
        self.wlr1 = nn.Parameter(torch.normal(mean = mean, std = std), requires_grad = True)
        self.wlr2 = nn.Parameter(torch.normal(mean = mean, std = std), requires_grad = True)
        self.blr1 = nn.Parameter(torch.tensor(0.0), requires_grad = True)

        # 输入门
        self.wpr1 = nn.Parameter(torch.normal(mean = mean, std = std), requires_grad = True)
        self.wpr2 = nn.Parameter(torch.normal(mean = mean, std = std), requires_grad = True)
        self.bpr1 = nn.Parameter(torch.tensor(0.0), requires_grad = True)
        self.wp1 = nn.Parameter(torch.normal(mean = mean, std = std), requires_grad = True)
        self.wp2 = nn.Parameter(torch.normal(mean = mean, std = std), requires_grad = True)
        self.bp1 = nn.Parameter(torch.tensor(0.0), requires_grad = True)

        # 输出门
        self.wo1 = nn.Parameter(torch.normal(mean = mean, std = std), requires_grad = True)
        self.wo2 = nn.Parameter(torch.normal(mean = mean, std = std), requires_grad = True)
        self.bo1 = nn.Parameter(torch.tensor(0.0), requires_grad = True)

    def lstm_unit(self, input, long_memory, short_memory):
        # 1 遗忘门
        long_remeber_percent = torch.sigmoid((short_memory * self.wlr1) + (input * self.wlr2) + self.blr1)

        # 2 输入门
        potential_remeber_percent = torch.sigmoid((short_memory * self.wpr1) + (input * self.wpr2) + self.bpr1)
        potential_memory = torch.tanh((short_memory * self.wp1) + (input * self.wp2) + self.bp1)
        updated_long_memory = ((long_memory * long_remeber_percent) + (potential_memory * potential_remeber_percent))

        # 3 输出门
        output_percent = torch.sigmoid((short_memory * self.wo1) + (input * self.wo2) + self.bo1)
        updated_short_memory = torch.tanh(updated_long_memory) * output_percent

        # 4 输出
        return ([updated_long_memory, updated_short_memory])

    def forward(self, input):
        long_memory = 0
        short_memory = 0

        for index in input:
            long_memory, short_memory = self.lstm_unit(input[index], long_memory, short_memory)

        return short_memory

    def configure_optimizers(self):
        return Adam(self.parameters())

    def training_step(self, batch, batch_index):
        input_i, label_i = batch
        output_i = self.forward(input_i[0])
        loss = (output_i - label_i) ** 2

        self.log("Train Loss", loss)

        if (label_i == 0):
            self.log("Out 0", output_i)
        else:
            self.log("Out 1", output_i)

        return loss
