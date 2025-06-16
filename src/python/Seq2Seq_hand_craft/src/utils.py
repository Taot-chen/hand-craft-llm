import torch
import unicodedata
import re
from io import open
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义开始符号和结束符号的常量
SOS_TOKEN = 0
EOS_TOKEN = 1

# 定义句子的最大长度
MAX_LENGTH = 128


# 语言类，负责处理语言相关操作，如创建词汇表
class Lang:
    def __init__(self, language):
        # 语言的名称
        self.language = language
        # 初始词汇表大小为2（SOS和EOS）
        self.size = 2
        # 单词到索引的映射
        self.word2index = {}
        # 索引到单词的映射,初始包含SOS和EOS
        self.index2word = {
            0: "SOS",
            1: "EOD"
        }

    # 添加单词到词汇表
    def add_word(self, word):
        if word not in self.word2index:
            # 为新单词分配索引
            self.word2index[word] = self.size
            # 保存索引到单词的映射
            self.index2word[self.size] = word
            # 增加词汇表的大小
            self.size += 1

    # 添加句子中的所有单词到词汇表
    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)


# 字符规范化
def unicode2ascii(text):
    # 将Unicode字符转换为ASCII字符, 同时去掉重音符号，将é转换为e
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')


# 确保输入的文本只包含标准的ASCII字符，防止多种编码格式导致不必要的错误
def normalizeString(text):
    s = unicode2ascii(text.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# 从文件中读取数据并进行初步处理
def readfile(file, input_lang, output_lang):
    data = open(file, encoding = 'utf-8').read().split('\n')
    # 将每行中的句子分割并规范化
    pairs = [[normalizeString(text) for text in line.split('\t')] for line in data]
    
    # 创建输入语言类, 输出语言类
    input_lang = Lang(input_lang)
    output_lang = Lang(output_lang)

    return input_lang, output_lang, pairs


# 数据过滤，定义只保留长度小于MAX_LENGTH的英语句子
def filtPair(pair):
    # handle the EOS_TOKEN and SOS_TOKEN
    return len(pair[0].split()) < MAX_LENGTH - 2 and len(pair[1].split()) < MAX_LENGTH - 2

def filtPairs(pairs):
    return [pair for pair in pairs if filtPair(pair)]


# 创建词汇表和句子过滤
def preprocess(file, input_lang, output_lang):
    input_lang, output_lang, pairs = readfile(file, input_lang, output_lang)
    pairs = filtPairs(pairs)
    # 添加输入/输出句子中的单词到词汇表
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    return input_lang, output_lang, pairs


# 将句子转换为张量
def sentence2tensor(Lang, sentence, device):
    indexs = [Lang.word2index[word] for word in sentence.split(' ')]
    indexs.append(EOS_TOKEN)
    # 返回形状为 (句子长度, 1) 的张量
    return torch.tensor(indexs, dtype = torch.long, device = device).view(-1, 1)


# 将句子对转换为张量表示
def pair2tensor(input_lang, output_lang, pair):
    input_tensor = sentence2tensor(input_lang, pair[0])
    output_tensor = sentence2tensor(output_lang, pair[1])
    return (input_tensor, output_tensor)

# plot loss
def plot_loss(epoch, loss, figpath = '../loss_curve.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(
        epoch,
        loss, 
        color='red',
        linestyle='-',
        # marker='o',
        linewidth=1,
        label='Training Loss'
    )
    plt.xlim(min(epoch), max(epoch))
    plt.ylim(0, max(loss)*1.1)
    plt.xlabel('epoch', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.title('Loss Curve', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(figpath)
