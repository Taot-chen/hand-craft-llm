hand-craft-llm

Try to start from first principles and manually craft content related to LLMs.


----------


## Transformer


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


## LLM


基于 pytorch 框架，从零开始搭建并训练一个最朴素的大语言模型。从数据集收集，到 Tokenizer 和 Model 训练，最终使用训练完成的模型尝试做文本生成。

后续考虑在现有的基础上添加 kv cache，扩展更大的词表，支持更大的上下文长度。并尝试在结构中引入更丰富的功能。

[从零搭建并训练一个大语言模型](https://github.com/Taot-chen/hand-craft-llm/blob/main/src/python/build_and_train_llm/README.md)



Reference：
* [tiny-universe](https://github.com/datawhalechina/tiny-universe/)
* [llama2.c](https://github.com/karpathy/llama2.c)




---------


## CNN

卷积神经网络(Convolutional Neural Networks, CNNs)是一种经典的深度学习算法，特别适用于图像处理和分析，适合用于图像特殊提取和分类。

从网络的原理和结构出发，了解 CNN 网络每一层的作用。基于 DNN 的反向传播算法，从第一性原理出发，来反推 CNN 的反向传播算法的的计算。

基于 pytorch 框架，从零开始 CNN 网络，并使用 MNIST 数据集来训练和评估网络。

[从零搭建并训练一个 CNN 网络](https://github.com/Taot-chen/hand-craft-llm/tree/main/src/python/CNN)


Reference：
* [CNN-MNIST-CPP-](https://github.com/xoslh/CNN-MNIST-CPP-)
* [PyTorch 卷积神经网络|菜鸟教程](https://www.runoob.com/pytorch/pytorch-cnn.html)




## RNN

循环神经网络（RNN）能够处理变长序列，擅长挖掘数据中的时序信息。但是存在长期以来问题，难以处理长序列中相距较远的信息关联。

从网络的原理和结构出发，了解 RNN 的构成。

利用 Pytorch 的 RNN module，以及手动搭建 RNN 网络，并训练评估对比了两种方式的效果。

[循环神经网络（RNN）实践](https://github.com/Taot-chen/hand-craft-llm/tree/main/src/python/RNN)



Reference：
* [hack-rnns](https://github.com/datawhalechina/hack-rnns/blob/main/docs/chapter1/chapter1.ipynb)
* [PyTorch 循环神经网络（RNN）|菜鸟教程](https://www.runoob.com/pytorch/pytorch-recurrent-neural-network.html)


