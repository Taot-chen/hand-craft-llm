hand-craft-llm

Try to start from first principles and manually craft content related to LLMs.


----------


## Build Transformer


目前，大部分大语言模型都是基于 Transformer 结构改进开发，Transformer 结构是 LLM 的基石。

基于论文[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)，使用 pytorch 框架从零开始搭建一个可运行的 Transformer 模型，以此来学习 Transformer 的原理。

[从零搭建 Transformer](https://github.com/Taot-chen/hand-craft-llm/blob/main/src/python/build_transformer/README.md)


Reference：
* [tiny-universe](https://github.com/datawhalechina/tiny-universe/)
* [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
* [NanoGPT](https://github.com/karpathy/nanoGPT)
* [ChineseNMT](https://github.com/hemingkx/ChineseNMT)
* [transformer-translator-pytorch](https://github.com/devjwsong/transformer-translator-pytorch)



----------


## Build and Train LLM


基于 pytorch 框架，从零开始搭建并训练一个最朴素的大语言模型。从数据集收集，到 Tokenizer 和 Model 训练，最终使用训练完成的模型尝试做文本生成。

后续考虑在现有的基础上添加 kv cache，扩展更大的词表，支持更大的上下文长度。并尝试在结构中引入更丰富的功能。

[从零搭建并训练一个大语言模型](https://github.com/Taot-chen/hand-craft-llm/blob/main/src/python/build_and_train_llm/README.md)



Reference：
* [tiny-universe](https://github.com/datawhalechina/tiny-universe/)
* [llama2.c](https://github.com/karpathy/llama2.c)
