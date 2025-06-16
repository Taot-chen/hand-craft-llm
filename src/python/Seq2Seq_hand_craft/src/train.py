import time
import math
from tqdm import tqdm
from torch import optim
import torch
import torch.nn as nn
import random
from utils import sentence2tensor, preprocess, device, MAX_LENGTH, plot_loss
from modeling_seq2seq import Encoder, AttentionDecoder, Seq2SeqModel

def train(input_tensor, output_tensor, model, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio=0.5):
    # 梯度归零
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    output_length = output_tensor.size(0)
    loss = 0

    # 前向传播
    decode_words = model(input_tensor, output_tensor, teacher_forcing_ratio)

    # 计算损失
    for index in range(output_length):
        target_word = output_tensor[index]
        output_word = decode_words[index].unsqueeze(0)  # 使维度变为 (1, output_size)
        loss += criterion(output_word, target_word)

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / output_length

# 训练迭代器
def train_iters(model, num_epoch, pairs, input_lang, output_lang, learning_rate = 0.01, plot_every = 100, teacher_forcing_ratio = 0.5, model_path = "../model_best.pt"):
    print(f"===================Train on {device}==================")
    start = time.time()
    plot_loss_total = 0
    total_loss = 0
    min_loss = 1e9
    epoch_list = []
    loss_list = []

    # 使用随机梯度下降优化器
    encoder_optimizer = optim.SGD(model.encoder.parameters(), lr = learning_rate)
    decoder_optimizer = optim.SGD(model.decoder.parameters(), lr = learning_rate)
    # 损失函数使用负对数似然损失
    criterion = nn.NLLLoss()

    tbar = tqdm(range(num_epoch), desc = 'epoch', leave = True)
    for epoch in tbar:
        # 随机选择一个句子对
        pair = random.choice(pairs)
        input_tensor = sentence2tensor(input_lang, pair[0], device)
        output_tensor = sentence2tensor(output_lang, pair[1], device)

        loss = train(input_tensor, output_tensor, model, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio)

        plot_loss_total += loss
        total_loss += loss
        epoch_list.append(epoch + 1)
        loss_list.append(total_loss / (epoch + 1))
        if epoch % plot_every:
            plot_loss_avg = plot_loss_total / plot_every
            plot_loss_total = 0
            tbar.set_postfix(loss = f"{total_loss / (epoch + 1): 4f}")
    plot_loss(epoch_list, loss_list)
    if total_loss / (epoch + 1) < min_loss:
        min_loss = total_loss / (epoch + 1)
        torch.save(model.state_dict(), model_path)

    

def launch_train():
    hidden_size = 256
    dropout =0.1
    num_epoch = 200000
    learning_rate = 0.015
    teacher_forcing_ratio = 0.5
    dataset_path = "../data/English-French.txt"
    input_lang, output_lang, pairs = preprocess(dataset_path, "eng", "fra")
    encoder = Encoder(input_lang.size, hidden_size, device = device).to(device)
    decoder = AttentionDecoder(hidden_size, output_lang.size, dropout = dropout, max_length = MAX_LENGTH, device = device).to(device)
    seq2seq_model = Seq2SeqModel(encoder, decoder, device = device).to(device)
    
    train_iters(seq2seq_model, num_epoch, pairs, input_lang, output_lang, learning_rate, plot_every = 1000, teacher_forcing_ratio = teacher_forcing_ratio)


if __name__ == "__main__":
    launch_train()

