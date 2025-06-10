import torch
import math
from torch.nn import functional as F
import torch.nn as nn
import inspect

# 注意力计算函数
def attention(q, k, v, dropout_module = None, is_causal = False, dropout = False, mask = None):
    # 计算 QK^T / sqrt(d_k)，维度为 (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

    if is_causal:
        attn = attn.masked_fill(mask[:, :, :k.size(-2), :k.size(-2)] == 0, float('-inf'))

    # 计算 softmax，维度为 (B, nh, T, T)
    attn = F.softmax(attn, dim = -1)
    if dropout_module is not None and dropout:
        attn = dropout_module(attn)
    
    # V * Score，维度为(B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    attn_output = attn @ v
    return attn_output


# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, config, is_causal = False):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        # Wq, Wk, Wv 参数矩阵，每个参数矩阵为 n_embd x n_embd
        self.attns = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd, bias = config.bias) for _ in range(3)])
        # 输出的线性层，维度为 n_embd x n_embd
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias = config.bias)
        # 注意力的 dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        # 残差连接的 dropout
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.is_causal = is_causal
        # 判断是否使用 Flash Attention，Pytorch 2.0 支持，即判断 torch.nn.functional.scaled_dot_product_attention 是否存在
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 使用 register_buffer 注册一个 bias 属性, bias 是一个上三角矩阵，维度为 1 x 1 x block_size x block_size，block_size 为序列最大长度
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1,
                    1, 
                    config.block_size,
                    config.block_size
                )
            )

    def forward(self, query, key, value):
        # 输入为 query、key、value，维度为 (B, T, n_embed)
        B, T, C = query.size()

        # 计算 Q、K、V，输入通过参数矩阵层，维度为 (B, T, n_embed) x (n_embed, n_embed) -> (B, T, n_embed)
        q, k, v = [self.attns[i](x) for i, x in zip(range(3), (query, key, value))]
        # 将 Q、K、V 拆分成多头，维度为 (B, T, n_head, C // n_head)，然后交换维度，变成 (B, n_head, T, C // n_head)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 注意力计算 
        if self.flash:
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p = self.dropout if self.training else 0,
                is_causal = self.is_causal
            )
        else:
            attn_out = attention(
                q,
                k,
                v,
                dropout_module = self.attn_dropout,
                is_causal = self.is_causal,
                dropout = self.dropout,
                mask = self.mask
            )

        # 将多头的结果拼接起来, 先交换维度为 (B, T, n_head, C // n_head)，再拼接成 (B, T, n_head * C // n_head)
        # 使用 contigonous() 函数保证内存是连续的，否则会报错
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)

        # 经过输出层计算，维度为 (B, T, C)，再经过线性层残差连接
        attn_out = self.o_proj(attn_out)
        attn_out = self.resid_dropout(attn_out)
        return attn_out


# 全连接模块
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Transformer 的全连接模块有两个线性层，中间加了一个 RELU 激活函数
        self.up_proj = nn.Linear(config.n_embd, 4 * config.n_embd, bias = config.bias)
        self.relu = nn.ReLU()
        self.down_proj = nn.Linear(config.n_embd * 4, config.n_embd, bias = config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        mlp_out = self.up_proj(x)
        mlp_out = self.relu(mlp_out)
        mlp_out = self.down_proj(mlp_out)
        mlp_out = self.dropout(mlp_out)
        return mlp_out



# 层归一化模块
class LayerNorm(nn.Module):
    def __init__(self, dim, bias = False, eps = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = eps

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)



# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 一个 Layer 中有两个 LayerNorm，分别在 Attention 之前和 MLP 之前
        self.ln1 = LayerNorm(config.n_embd, bias=config.bias)
        # Encoder 不需要掩码，传入 is_causal=False
        self.attn = MultiHeadAttention(config, is_causal=False)
        self.ln2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, input):
        output = self.ln1(input)
        output = output + self.attn(output, output, output)
        output = self.ln2(output)
        output = output + self.mlp(output)
        return output


# Encoder Module
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        # 一个 Encoder 由 N 个 Encoder Layer 组成
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.n_layer)])
        self.norm = LayerNorm(config.n_embd, bias = config.bias)

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return self.norm(input)


# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 一个 Layer 中有三个 LayerNorm，分别在 Mask Attention 之前、Self Attention 之前和 MLP 之前
        self.ln1 = LayerNorm(config.n_embd, bias=config.bias)
        # Decoder 的第一个部分是 Mask Attention，传入 is_causal=True
        self.mask_attn = MultiHeadAttention(config, is_causal = True)
        self.ln2 = LayerNorm(config.n_embd, bias = config.bias)
        # Decoder 的第二个部分是 类似于 Encoder 的 Attention，传入 is_causal=False
        self.attn = MultiHeadAttention(config, is_causal = False)
        self.ln3 = LayerNorm(config.n_embd, bias = config.bias)
        self.mlp = MLP(config)

    def forward(self, input, encode_out):
        output = self.ln1(input)
        # 第一部分是一个 Mask Self Attention，Q、K、V 都是 output
        output = output + self.mask_attn(output, output, output)
        output = self.ln2(output)
        # 第二部分是一个类似于 Encoder 的 Attention，Q 是 output，K、V 是 Encoder 的输出
        output = output + self.attn(output, encode_out, encode_out)
        output = self.ln3(output)
        output = output + self.mlp(output)
        return output


# Decoder Module
class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__() 
        # 一个 Decoder 由 N 个 Decoder Layer 组成
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layer)])
        self.norm = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, input, encode_out):
        for layer in self.layers:
            input = layer(input, encode_out)
        output = self.norm(input)
        return output



# Position Embedding
class PositionEmbedding(nn.Module):
    def __init__(self, config, n = 10000):
        super(PositionEmbedding, self).__init__()
        self.dropout = nn.Dropout(p = config.dropout)

        # block size 是序列的最大长度
        pe = torch.zeros(config.block_size, config.n_embd)
        position = torch.arange(0, config.block_size).unsqueeze(1)
        div_item = torch.exp(
            torch.arange(0, config.n_embd,  2) * (-1) * (math.log(n) / config.n_embd)
        )
        pe[:, 0::2] = torch.sin(position * div_item)
        pe[:, 1::2] = torch.cos(position * div_item)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, input):
        input = input + self.pe[:, : input.size(1)].requires_grad_(False)
        output = self.dropout(input)
        return output


# Transformer Model
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                embedding = nn.Embedding(config.vocab_size, config.n_embd),
                position_embedding = PositionEmbedding(config),
                dropout = nn.Dropout(config.dropout),
                encoder = Encoder(config),
                decoder = Decoder(config)
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        # 初始化所有的权重
        self.apply(self._init_weights)
        # 查看所有参数的数量
        print("number of parameters: %.2fM parameters" % (self.get_num_params() / (1024 ** 2),))

    def get_num_params(self, non_embedding = False):
        # non_embedding: 不统计 embedding 的参数
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.embedding.weight.numel()
        return n_params

    def _init_weights(self, module):
        # 线性层和 Embedding 层初始化为正则分布
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self, input, targets = None):
        # input，维度为 (batch size, sequence length)
        # targets 为目标序列，用于计算 loss
        device = input.device
        print(f"Run on {device}")
        b, t = input.shape
        assert t <= self.config.block_size, f"不能计算该序列，该序列长度为 {t}, 最大序列长度只有 {self.config.block_size}"

        # 将输入 input 通过 Embedding 层，期望得到维度为 (batch size, sequence length, n_embd)
        tok_emb = self.transformer.embedding(input)
        print("tok_emb",tok_emb.size())
    
        # 通过位置编码
        pos_emb = self.transformer.position_embedding(tok_emb) 
        # 再进行 Dropout
        x = self.transformer.dropout(pos_emb)
        print("x after position_embedding:",x.size())

        # 通过 Encoder
        enc_out = self.transformer.encoder(x)
        print("enc_out:",enc_out.size())

        # 通过 Decoder
        x = self.transformer.decoder(x, enc_out)
        print("x after decoder:",x.size())

        if targets is not None:
            # 训练阶段，如果给了 targets，就计算 loss
            # 先通过最后的 Linear 层，得到维度为 (batch size, sequence length, vocab size)
            logits = self.lm_head(x)

            # 再跟 targets 计算交叉熵
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推理阶段，只需要 logits，loss 为 None
            # 取 -1 是只取序列中的最后一个作为输出
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    '''配置优化器'''
    # weight_decay: 权重衰减系数，learning_rate: 学习率，betas: AdamW 的 betas，device_type: 设备类型
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}

        # 过滤掉不需要更新的参数
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # 参数根据维度分为两组。
        # 维度大于等于2的参数（通常是权重）会应用权重衰减，而维度小于2的参数（通常是偏置和层归一化参数）不会应用权重衰减。
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # 打印一下参数数量
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"应用权重衰减的层数: {len(decay_params)}； 总参数量为：{num_decay_params:,}")
        print(f"不应用权重衰减的层数: {len(nodecay_params)}, 总参数量为：{num_nodecay_params:,}")

        # 检查 torch.optim.AdamW 是否支持融合版本（fused version），这是针对 CUDA 设备优化的版本。如果可用且 device_type 为 'cuda'，则使用融合版本。
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        # 创建优化器
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"使用 fused AdamW: {use_fused}")

        return optimizer


    '''推理生成'''
    @torch.no_grad()
    def generate(self, input, max_new_tokens, temperature=1.0, top_k=None):
        # 推理阶段，输入为 input，维度为 (batch size, sequence length)，max_new_tokens 为最大生成的 token 数量
        for _ in range(max_new_tokens):
            # 如果输入序列太长，将它截断到 block_size
            input_cond = input if input.size(1) <= self.config.block_size else input[:, -self.config.block_size:]

            # 前向计算，得到 logits，维度为 (batch size, sequence length, vocab size)
            logits, _ = self(input_cond)
            # 使用最后一个 token 的 logits 作为当前输出，除以温度系数控制其多样性
            logits = logits[:, -1, :] / temperature

            # 如果使用 Top K 采样，将 logits 中除了 top_k 个元素的概率置为 0
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # 对输出结果进行 Softmax
            probs = F.softmax(logits, dim=-1)
            # 对结果概率进行采样
            input_next = torch.multinomial(probs, num_samples=1)

            # 将输出结果拼接到输入序列后面，作为下一次的输入
            input = torch.cat((input, input_next), dim=1)
        return input
