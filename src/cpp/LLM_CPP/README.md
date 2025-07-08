# LLM with CPP

> 参考 [llm.c](https://github.com/karpathy/llm.c) ，使用纯粹的 C++ 和 CUDA 来实现大语言模型的训练、推理和部署。

当前大语言模型的训练、推理和部署，都有非常好用的框架，例如 Deepspeed/vLLM 等。这些框架都强依赖 pytorch。为了能够更深入学习大语言模型的原理，以及训练、推理和部署过程中的难点和优化点，便有了这个项目。

本项目以学习为主，在此过程中，使用纯粹的 C++ 和 CUDA 来实现大语言模型的训练、推理和部署，并同时使用 pytorch 实现相同的效果，作为参考基线。

本仓库的内容：

* 参考 [llm.c](https://github.com/karpathy/llm.c) ，逐步复现如下内容：
    * 复现 GPT-2 (124M) 模型，参考：https://github.com/karpathy/llm.c/blob/master/scripts/README.md，可以使用下面的方式：
    * 单 GPU，fp32
        ```bash
        chmod +x dev/download_starter_pack.sh
        ./dev/download_starter_pack.sh
        make train_gpt2fp32cu
        ./train_gpt2fp32cu
        ```
        `download_starter_pack.sh` 脚本会下载一堆 `.bin` 文件，这些文件包含：

        * 以 fp32 和 bfloat16 格式保存的 GPT-2 124M 模型
        * 用于单元测试的“调试状态”（一小批数据，以及目标激活值和梯度）
        * GPT-2 分词器（tokenizer）
        * 分词处理后的 tinyshakespeare 数据集
    或者，不运行 `.sh` 脚本，可以手动重新创建这些文件，如下所示：

    ```bash
    pip install -r requirements.txt
    python dev/data/tinyshakespeare.py
    python train_gpt2.py
    ```

    * CPU 版本

    在不使用 GPU 的情况下，只使用CPU也可以，但是没法走的太远。这是一个完全的 C 语言实现（great）。

    不是从头开始训练，而是可以微调（finetune）一个 GPT-2 small (124M) 模型来输出类似莎士比亚的文本：

    ```bash
    chmod u+x ./dev/download_starter_pack.sh
    ./dev/download_starter_pack.sh  # 下载一个已经分词的 tinyshakespeare 数据集并下载 GPT-2 (124M) 的权重
    make train_gpt2
    OMP_NUM_THREADS=8 ./train_gpt2  # 在 C 语言中从这些权重初始化，并在 tinyshakespeare 上用 AdamW 训练 40 步（使用批次大小 4，上下文长度仅 64），评估验证损失，并采样一些文本
    ```

    当然，也可以不使用脚本，可以通过运行 `python dev/data/tinyshakespeare.py` 然后 `python train_gpt2.py` 来重现完全相同的 `.bin` 文件和构件

    * 数据集

    `/dev/data/(dataset).py` 内部的数据文件负责下载数据集、分词（tokenizing）并将分词（tokens）保存到 `.bin` 文件中，这些文件可以很容易地从 C 语言中读取。例如，当运行：

    ```bash
    python dev/data/tinyshakespeare.py
    ```

    会下载并分词 [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)数据集，脚本输出类似：

    ```bash
    writing 32,768 tokens to ./dev/data/tinyshakespeare/tiny_shakespeare_val.bin
    writing 305,260 tokens to ./dev/data/tinyshakespeare/tiny_shakespeare_train.bin
    ```

    `.bin` 文件包含一个简短的头部（1024 字节），然后是一个 `uint16` 类型的分词（token）流，指示使用 GPT-2 分词器得到的分词 ID（token ids）。更多数据集可在 `/dev/data` 中找到。


    * 测试

    附加了一个简单的单元测试，用于确保我们的 C 代码与 PyTorch 代码一致。

    以 CPU 为例，编译并运行：

    ```bash
    make test_gpt2
    ./test_gpt2
    ```

    这会加载 `train_gpt2.py` 写入的 `gpt2_124M_debug_state.bin` 文件，运行一次前向传播（forward pass），将 `logits` 和 `loss` 与 PyTorch 参考实现进行比较，然后用 Adam 进行 10 次训练迭代，并确保损失与 PyTorch 匹配。

    测试 GPU 版本：
    
    ```bash
    # fp32 测试 (不支持 cudnn)
    make test_gpt2cu PRECISION=FP32 && ./test_gpt2cu

    # 混合精度 cudnn 测试
    make test_gpt2cu USE_CUDNN=1 && ./test_gpt2cu
    ```

    这测试了 `fp32` 路径和混合精度路径。测试通过会打印`overall okay: 1`.


    * get_started

        在 `doc/layernorm/layernorm.md` 附加了一个非常小的教程。这是一个简单、分步的指南，用于实现 GPT-2 模型的单个层——`层归一化`（layernorm）层。这是理解层如何在 C 语言中实现的良好起点。

    * ​​Flash Attention​​

        截至 2024 年 5 月 1 日，使用 cuDNN 中的 Flash Attention。因为 cuDNN 会将编译时间从几秒增加到约一分钟，并且此代码路径目前非常新，所以默认情况下是禁用的。您可以通过如下方式编译启用它：

        ```bash
        make train_gpt2cu USE_CUDNN=1
        ```

        这将尝试用 cudnn 编译并运行它。您必须在系统上安装 cuDNN。

        使用 `apt-get` 的 cuDNN 安装说明将获取默认的 cuDNN 包集。对于最小化设置，cuDNN 开发包就足够了，例如在 Ubuntu 22.04 上安装 CUDA 12.x：

        ```bash
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt-get update
        sudo apt-get -y install libcudnn9-dev-cuda-12
        ```

        除此之外，您还需要 cuDNN 前端（frontend），但这只是头文件。只需将仓库克隆到您的磁盘即可。Makefile 当前会在您的主目录或当前目录中查找它。如果您把它放在其他地方，请在 `make` 命令行中添加 `CUDNN_FRONTEND_PATH=/path/to/your/cudnn-frontend/include`。

    * 多 GPU 训练 (Multi-GPU Training)​

        确保安装 MPI 和 NCCL，例如在 Linux 上：

        ```bash
        sudo apt install openmpi-bin openmpi-doc libopenmpi-dev
        ```

        编译：

        ```bash
        make train_gpt2cu
        mpirun -np <GPU 数量> ./train_gpt2cu
        ```

    
    * 多节点训练 (Multi-Node Training)​

        确保已按照多 GPU 部分的说明安装了 NCCL。

        目前支持三种方式进行多节点训练：

        * ​​使用 OpenMPI 交换 nccl id 并初始化 NCCL。​​ 详情请参见 `./scripts/multi_node/run_gpt2_124M_mpi.sh` 脚本。
        * ​​使用共享文件系统初始化 NCCL。​​ 详情请参见 `./scripts/multi_node/run_gpt2_124M_fs.sbatch` 脚本。
        * ​​使用 TCP 套接字初始化 NCCL。​​ 详情请参见 `./scripts/multi_node/run_gpt2_124M_tcp.sbatch` 脚本。

        ​​注意：​​ 如果在 `slurm` 环境中运行，并且您的 `slurm` 不支持 PMIx（考虑到 `slurm-wlm` 放弃了对 `PMIx` 的支持，我们认为这将是一种常见情况），您将不得不使用 FS (2) 或 TCP (3) 方法。

        要测试您的 slurm 是否支持 PMIx，请运行：

        ```bash
        srun --mpi=list
        ```

        查看输出中是否有 pmix。如果您没有设置 slurm，可以使用 `mpirun` 启动多节点运行 - MPI (1)。


    * 实验 / 参数扫描 (Experiments / Sweeps)​

        在配备 4 个 GPU 的机器上对 `TinyStories` 数据集扫描学习率（learning rate）的示例过程。运行一个 shell 脚本 `sweep.sh`（当然，您需要先 `chmod u+x sweep.sh`）：

        ```bash
        #!/bin/bash
        learning_rates=(3e-5 1e-4 3e-4 1e-3)
        for i in {0..3}; do
            export CUDA_VISIBLE_DEVICES=$i
            screen -dmS "tr$i" bash -c "./train_gpt2cu -i data/TinyStories -v 250 -s 250 -g 144 -l ${learning_rates[$i]} -o stories$i.log"
        done
        # 您可以用以下命令关闭它们
        # screen -ls | grep -E "tr[0-3]" | cut -d. -f1 | xargs -I {} screen -X -S {} quit
        ```

        此示例打开 4 个 screen 会话，并使用不同的学习率运行四个命令。这将把包含所有损失的日志文件写入 `stories$i.log`，您可以在 Python 中随意绘制图表。`dev/vislog.ipynb` 中有一个关于如何解析和绘制这些日志文件的快速示例。



* 在复现的的基础上，对项目的实现有了一定的认识，后面尝试将该项目拓展成轻量化的训练、推理、部署的框架，并使用 qwen3 等模型来验证。

