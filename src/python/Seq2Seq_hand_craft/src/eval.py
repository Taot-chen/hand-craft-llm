import random
import torch
from utils import sentence2tensor, preprocess, device, MAX_LENGTH, SOS_TOKEN, EOS_TOKEN
from modeling_seq2seq import Encoder, AttentionDecoder

def evaluate(encoder, decoder, input_lang, output_lang, sentence, max_length=MAX_LENGTH):
    with torch.no_grad(): 
        input_tensor = sentence2tensor(input_lang, sentence, device).to(device)
        input_length = input_tensor.size(0)

        # 初始化编码器隐藏状态
        encoder_hidden = encoder.init_hidden()
        # 初始化编码器输出
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        # 编码器过程
        for index in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[index], encoder_hidden
            )
            encoder_outputs[index] = encoder_output[0, 0]

        # 解码器的输入初始化为SOS符号
        decoder_input = torch.tensor([[SOS_TOKEN]], device=device)
        # 解码器的隐藏状态初始化为编码器的隐藏状态
        decoder_hidden = encoder_hidden

        decode_words = []
        decoder_attns = torch.zeros(max_length, max_length, device = device)

        # 解码器过程
        index = 0
        for index in range(max_length):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_attns[index] = decoder_attn.data
            # 获取概率最高的单词索引
            topv, topi = decoder_output.topk(1)
            # 如果是结束符，停止解码
            if topi.item() == EOS_TOKEN:
                decode_words.append("<EOS>")
                break
            else:
                decode_words.append(output_lang.index2word[topi.item()])

            # # 将预测的单词作为下一个时间步的输入
            decoder_input = topi.squeeze().detach()
        return decode_words, decoder_attns[: index + 1]


def launch_eval(encoder, decoder, input_lang, output_lang, pairs, num_eval = 5):
    print(f"===================Eval on {device}==================")
    for index in range(num_eval):
        pair = random.choice(pairs)
        print(f'输入> {pair[0]}')
        print(f'输出> {pair[1]}')
        # 评估模型的输出
        output_words, output_attn = evaluate(encoder, decoder, input_lang, output_lang, pair[0])
        # 将生成的单词拼接为句子
        output_sentence = ''.join(output_words)
        print(f'预测结果> {output_sentence}')
        print('-' * 50)

if __name__ == "__main__":
    hidden_size = 256
    dropout =0.1
    num_eval = 5
    dataset_path = "../data/English-French.txt"
    input_lang, output_lang, pairs = preprocess(dataset_path, "eng", "fra")
    encoder = Encoder(input_lang.size, hidden_size, device = device).to(device)
    decoder = AttentionDecoder(hidden_size, output_lang.size, dropout = dropout, max_length = MAX_LENGTH, device = device).to(device)
    launch_eval(encoder, decoder, input_lang, output_lang, pairs, num_eval = num_eval)
