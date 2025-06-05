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


