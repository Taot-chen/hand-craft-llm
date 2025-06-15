import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import lightning as L
from torch.utils.data import TensorDataset, DataLoader


class CustomLSTM(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size = 1, hidden_size = 1)

    def forward(self, input):
        input_trans = input.view(len(input), 1)

        lstm_out, tmp = self.lstm(input_trans)
        prediction = lstm_out[-1]

        return prediction

    def configure_optimizers(self):
        return Adam(self.parameters(), lr = 0.1)

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

