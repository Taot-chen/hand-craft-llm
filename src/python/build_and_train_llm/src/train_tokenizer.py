import glob
import json
import os
from tqdm import tqdm
import sentencepiece as spm
import argparse
from pathlib import Path

def load_text_from_files(path):
    path_list = glob.glob(path)
    text_data = []
    for file_path in path_list:
        with open(file_path, 'r', encoding='utf-8') as file:
            text_data.extend(file.readlines())
    return text_data

def batch_iterator(text_data, batch_size=1024):
    for i in range(0, len(text_data), batch_size):
        yield text_data[i:i + batch_size]

def train_vocab(vocab_size: int=32000, num_shards: int=32, dataset_dir:str =""):
    """
    vocab_size: int, 词汇表的大小，决定分词器的词汇量。
    num_shards: int, 用于加快词汇表训练的效率，指定要处理的分片数量。
    """
    assert vocab_size > 0, "Vocab size must be positive"

    # SentencePiece 模型的前缀路径，将用于保存分词器
    curr_dir = Path(__file__).parent.resolve()  
    prefix = os.path.join(curr_dir, f"tok{vocab_size}")

    # 1) 将多个分片中的文本导出为单个文本文件 tiny.txt
    tiny_file = os.path.join(dataset_dir, "tiny.txt")
    shard_filenames = sorted(glob.glob(os.path.join(dataset_dir, "*.json")))

    # 创建 tiny.txt 文件并写入指定数量的分片中的文本
    print(f"Writing temporary file {tiny_file} with {num_shards} shards...")
    with open(tiny_file, "w", encoding="utf-8") as of:
        # 遍历前 num_shards 个分片
        for shard in tqdm(shard_filenames[:num_shards]):
            with open(shard, "r") as f:
                data = json.load(f)  # 读取分片中的JSON数据
            # 遍历每个例子，将其中的故事文本写入 tiny.txt 文件
            for item in data:
                text = item["story"]
                text = text.strip()  # 去除文本首尾的空白字符
                of.write(text + "\n")  # 每个文本写入一行

    # 输出生成的 tiny.txt 文件的大小
    print(f"Size is: {os.path.getsize(tiny_file) / 1024 / 1024:.2f} MB")

    # 2) 使用 SentencePiece 训练分词器
    print("Training the vocab...")
    spm.SentencePieceTrainer.train(
        input=tiny_file,         # 输入文件为之前生成的 tiny.txt
        model_prefix=prefix,     # 模型前缀路径
        model_type="bpe",        # 使用 Byte-Pair Encoding (BPE) 训练分词器
        vocab_size=vocab_size,   # 词汇表大小
        self_test_sample_size=0, # 自测样本大小设置为 0
        input_format="text",     # 输入文件格式为纯文本
        character_coverage=1.0,  # 覆盖所有字符（包括非常见字符）
        num_threads=os.cpu_count(),  # 使用 CPU 的线程数
        split_digits=True,       # 拆分数字
        allow_whitespace_only_pieces=True,  # 允许仅由空格组成的词元
        byte_fallback=True,      # 启用字节级回退
        unk_surface=r" \342\201\207 ",  # UNK token 表示未知字符的方式
        normalization_rule_name="identity"  # 使用“identity”归一化规则
    )

    # # 3) 可选的清理操作，询问用户是否删除临时文件 tiny.txt
    # dec = input(f"Delete the temporary file {tiny_file}? [y/N] ")
    # if dec.lower() == "y":
    #     os.remove(tiny_file)  # 删除临时文件
    #     print(f"Deleted {tiny_file}")

    # 输出模型保存的路径
    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=4096, help="vocab size")
    parser.add_argument("--dataset_dir", type=str, default="", help="dataset path")
    parser.add_argument("--num_shards", type=int, default=32, help="number of shards")
    args = parser.parse_args()

    train_vocab(args.vocab_size, args.num_shards, args.dataset_dir)
