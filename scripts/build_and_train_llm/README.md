## 从零搭建并训练一个大语言模型

Reference：
* [tiny-universe](https://github.com/datawhalechina/tiny-universe/)
* [llama2.c](https://github.com/karpathy/llama2.c)

从零开始搭建并训练一个大语言模型主要工作：

* 训练分词器
* 数据预处理
* 训练模型
* 部署推理




### 1 训练 Tokenizer（分词器）

Tokenizer 的作用是做分词，将文本转换为数字序列，以便模型能够理解和处理。

训练 Tokenizer 使用的数据集是 [TinyStories](https://www.modelscope.cn/datasets/AI-ModelScope/TinyStories)，这是一个由GPT-3.5和GPT-4生成的小型故事数据集，包含简短的故事，且词汇量有限。

在这里，采用字符级Tokenizer，将文本中的每个字符映射为对应的数字。


#### 1.1 数据集获取

* 数据集下载：

`./download_dataset.sh`

或者直接运行命令：

```bash
modelscope download --dataset AI-ModelScope/TinyStories --local_dir ./TinyStories
```

* 解压数据集

```bash
cd TinyStories && tar -zxvf TinyStories_all_data.tar.gz
mkdir dataset
tar -zxvf TinyStories_all_data.tar.gz -C ./dataset/
```

由于 TinyStory 数据集较小，词汇量有限，将词表大小设置为 4,096。训练完成后，得到的 Tokenizer 能够将文本转换为数字序列，也可以将数字序列还原为文本。



#### 1.2 训练 Tokenizer

```bash
python3 train_tokenizer.py --vocab_size=4096 --dataset_dir=./TinyStories/TinyStories_all_data/
```

在这里，使用了 `SentencePiece` 库来训练自定义的 Tokenizer。首先，需要从 TinyStory 数据集中提取文本内容，作为训练的输入数据。SentencePiece 是一种基于子词单元的分词算法，能够有效处理不同语言中的词汇碎片化问题。

训练结束之后，会在当前路径下(`hand-craft-llm/scripts/build_and_train_llm/src`)生成`tok4096.model` 和 `tok4096.vocab`，其中 `tok4096.model`` 是训练好的模型文件。这个文件可以用于将文本数据转换为 Token 序列，也可以将 Token 序列还原为文本。


在 `tokenizer.py` 文件中定义了一个 Tokenizer 类。这个类封装了 Tokenizer 的常用操作，例如文本编码和解码功能，并支持加载训练好的模型文件。通过这个类，可以轻松地将文本转换为模型可接受的数字序列，或将预测结果转化为可读的文本。


在这个 Tokenizer 类中，首先初始化了一些特殊的 token ID，这些特殊 tokens 分别用于填充、处理未识别的词汇、表示句子的开头和结尾等。在模型训练和推理过程中，正确处理这些特殊 tokens 对于提升模型性能很重要。

定义了两个常用的方法：

* `encode` 方法：该方法负责将输入文本转换为 token ID 序列。通过加载预训练的 Tokenizer 模型，可以对文本进行分词，将其拆解为词或子词，并将其映射为相应的数字表示。这个数字序列可以被模型接受用于训练和推理。

* `decode` 方法：与 `encode` 方法相反，`decode` 方法用于将 token ID 序列还原为可读的文本。它将数字序列转换回对应的 tokens，并拼接成完整的文本，从而可以对模型的输出进行解释和展示。

测试 Tokenizer 的功能，验证其是否能够正确地将文本转换为数字序列，或者将数字序列还原为文本:

`test_tokenizer.py`

```python
from tokenizer import Tokenizer

def main():
    enc = Tokenizer('./tok4096.model')
    text = 'Hello, world!'
    print(enc.encode(text, bos=True, eos=True))
    print(enc.decode(enc.encode(text, bos=True, eos=True)))

if __name__ == "__main__":
    main()

# OUTPUT:
# [1, 346, 2233, 4010, 1475, 4021, 2]
# Hello, world!
```


### 2 数据预处理

在文件 `preprocess.py` 中定义了 `process_shard ` 函数，用于处理数据分片。该函数的主要功能是将文本数据分词后，转换为更高效的二进制文件格式，以便后续更快速地加载和处理数据。定义了 `pretokenize` 函数，用于批量处理多个数据分片。通过这一函数，所有数据可以并行处理，进一步加快预处理的速度。

设计了一个 `PretokDataset` 类，用于加载已预处理好的数据集。继承自 `torch.utils.data.IterableDataset` 来定义该数据集，这使得可以更灵活、高效地处理数据。在这个类中，核心是 `__iter__` 方法，它负责生成用于训练的数据批次。

最后，定义了一个 `Task` 类，专门用于迭代数据集，并生成模型所需的输入和目标输出。这一部分的设计确保了数据流的顺畅对接，为模型训练提供了标准化的数据输入。

数据预处理命令：

```bash
python3 preprocess.py --vocab_size=4096 --dataset_dir=./TinyStories/TinyStories_all_data/ --tokenizer_model_path=./tok4096.model
```

数据预处理完成之后，在`TinyStories/TinyStories_all_data/tok4096/`下面会有预处理完成的 `datann.bin`。



### 3 训练模型

在数据预处理完成后，就可以开始训练模型了。

使用的模型和 LLama2 结构相同， Decoder only Transformer 模型，使用 Pytorch 实现。建模代码在`modeling_tinyllm.py`中。在建模脚本中，`generate` 方法展示了模型如何基于已有的上下文生成后续 token 的机制。

```python
@torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        给定输入序列 idx（形状为 (bz,seq_len) 的长整型张量），通过多次生成新 token 来完成序列。
        在 model.eval() 模式下运行。效率较低的采样版本，没有使用 kv cache。
        """

        for _ in range(max_new_tokens):
            # 如果序列上下文过长，截断它到最大长度
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]

            # 前向传播获取序列中最后一个位置的 logits
            logits = self(idx_cond)
            logits = logits[:, -1, :] # 只保留最后一个时间步的输出

            if temperature == 0.0:
                # 选择最有可能的索引
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # 缩放 logits 并应用 softmax
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            # 将采样的索引添加到序列中并继续
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
```

在 `generate` 方法中，首先获取序列中最后一个位置的 `logits`，然后基于这些 `logits` 生成新的 `token`。接着，生成的新 `token` 会被添加到序列中，模型随后会继续生成下一个 `token`。通过这种迭代过程，能够生成完整的文本。

`train_model.py` 中定义了很多超参数，包括但不限于模型的维度，层数，学习率等。`python3 train_model.py`命令即可开始训练模型。

```python
# -----------------------------------------------------------------------------
# I/O 配置，用于定义输出目录和训练时的日志记录与评估设置
out_dir = "./"  # 模型输出保存路径
eval_interval = 2000  # 评估间隔步数
log_interval = 1  # 日志记录间隔步数
eval_iters = 100  # 每次评估时迭代的步数
eval_only = False  # 如果为True，脚本在第一次评估后立即退出
always_save_checkpoint = False  # 如果为True，在每次评估后总是保存检查点
init_from = "scratch"  # 可以选择从头开始训练（'scratch'）或从已有的检查点恢复（'resume'）

# 数据配置
batch_size = 8  # 每个微批次的样本数量，如果使用梯度累积，实际批次大小将更大
max_seq_len = 256  # 最大序列长度
vocab_size = 4096  # 自定义词汇表大小

# 模型配置
dim = 288  # 模型的隐藏层维度
n_layers = 8  # Transformer的层数
n_heads = 8  # 注意力头的数量
n_kv_heads = 4  # 模型分组
multiple_of = 32  # 在某些层的维度必须是该数的倍数
dropout = 0.0  # Dropout概率

# AdamW优化器配置
gradient_accumulation_steps = 4  # 梯度累积步数，用于模拟更大的批次
learning_rate = 5e-4  # 最大学习率
max_iters = 100000  # 总的训练迭代次数
weight_decay = 1e-1  # 权重衰减系数
beta1 = 0.9  # AdamW优化器的β1参数
beta2 = 0.95  # AdamW优化器的β2参数
grad_clip = 1.0  # 梯度裁剪阈值，0表示不裁剪

# 学习率衰减配置
decay_lr = True  # 是否启用学习率衰减
warmup_iters = 1000  # 学习率预热的步数

# 系统设置
device = "cuda:0"  # 设备选择：'cpu'，'cuda'，'cuda:0'等
dtype = "bfloat16"  # 数据类型：'float32'，'bfloat16'，'float16'
```

训练过程中显存消耗在 3GB 以内：

![alt text](./images/image.png)


