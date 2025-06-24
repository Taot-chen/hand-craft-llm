## DeepSeek 推理

在大模型领域，*推理*这个中文单词有2个含义，

* 一个是*inference*，指对输入的数据进行处理并生成输出结果的过程；
* 另一个是*reasoning*，指大模型在运行过程中进行思考，并运用各种和逻辑方法来得出结论。


这里的推理的是*inference*，即如何处理输入数据并生成输出结果。大模型的推理引擎有很多，并且都各有特色，一些常见的LLM推理引擎：

* Transformers（Hugging Face推出的库，适合实验和学习）
* vLLM（一个**高效的推理加速框架**）
* SGLang（适合在各种场景下定制推理方式，需要一定技术基础）
* Llama.cpp（纯C/C++实现，不需要安装额外依赖环境，同时支持多种除CUDA以外的GPU调用方式，需要一定技术基础）
* Ollama（**对Llama.cpp的封装**，使用起来非常简单，对小白友好）
* MLX（专门为苹果芯片优化的机器学习框架）
* Xinference（对vLLM、SGLang、Llama.cpp、Transformers、MLX的封装，特点是部署快捷、使用简单）
* LMDeploy（一个吞吐量比vLLM高的推理加速框架）

常用的推理加速手段:

### 1 KV Cache

大模型的生成过程是自回归式的，每次输出一个新的 token 并拼接入序列，反复迭代直到结束。在这推理过程中有一个步骤是计算自注意力，对每个输入 token 计算其对应的 Q(Query)、K(Key) 和 V(Value)，并计算注意力分数：

$$
\text{Attention} (Q, K, V) = \text{softmax} (\frac{QK^T}{\sqrt{d_k}})V
$$

每次自注意力的计算都有大量的重复内容，可以将计算结果保存下来留着下次使用，而这就是 KV Cache。


对于多头潜在注意力(MLA)，可以考虑如何压缩 KVCache 的显存占用。


### 2 Persistent Batch（也叫continuous batching）

用一个模型同时推理多个序列，增加模型吞吐量。

具体原理后面在 vllm 里面看。


### 3 KV Cache 复用

在大模型的 API 使用场景中，用户的输入有很大的比例都是重复的，例如用户的 prompt 中会有很多重复的引用部分；在多轮对话中，每一轮都需要将前几轮的内容重复输入。

鉴于此，就有了 kv cache 缓存技术，秉持着优先缓存在显存，显存不够内存来凑，内存不够硬盘来凑，硬盘不够分布式云存储来凑的原则，把未来会重复使用的内容，缓存在各种存储介质中。如果存在重复的输入，那么重复的部分直接从缓存读取，无需计算。该技术不仅能够降低推理延迟，还能够有效降低推理成本。

kv cache 复用的方法可以参考SGLang的RadixAttention，最核心的思想就是具有相同前缀的输入可以共享KV Cache。
SGLang论文：https://arxiv.org/abs/2312.07104
另外，在 vllm 和 NVIDIA dynamo 中都有使用这项技术。


### 4 量化

* GPTQ，AWQ 等经典的量化技术。

* kv cache 量化

* 1.58bit 的 BitNet。


### 5 MoE 架构通过减少激活值来加速计算



