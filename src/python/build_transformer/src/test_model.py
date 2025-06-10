from modeling_tiny_transformer import Transformer
from configuration_tiny_transformer import TransformerConfig
import torch

def get_model_cpu():
    model_config = TransformerConfig(
        block_size = 128,
        vocab_size = 32,
        n_layer = 4,
        n_head = 4,
        n_embd = 768,
        dropout = 0.0,
        bias = True
    )

    # print("model_config: \n", model_config)
    model = Transformer(model_config)
    # print("model:\n", model)

    # 前向传播
    input = torch.randint(1, 10, (4, 8))
    logits, _ = model(input)
    print("input: ", input.size())
    print("logits: ", logits.size())
    """
    输出：
    number of parameters: 63.14M parameters
    Run on cpu
    tok_emb torch.Size([4, 8, 768])
    x after position_embedding: torch.Size([4, 8, 768])
    enc_out: torch.Size([4, 8, 768])
    x after decoder: torch.Size([4, 8, 768])
    input:  torch.Size([4, 8])
    logits:  torch.Size([4, 1, 32])
    """

    # 推理
    result = model.generate(input, 3)
    print("Generate result: ", result.size())
    """
    输出：
    Run on cpu
    tok_emb torch.Size([4, 8, 768])
    x after position_embedding: torch.Size([4, 8, 768])
    enc_out: torch.Size([4, 8, 768])
    x after decoder: torch.Size([4, 8, 768])
    Run on cpu
    tok_emb torch.Size([4, 9, 768])
    x after position_embedding: torch.Size([4, 9, 768])
    enc_out: torch.Size([4, 9, 768])
    x after decoder: torch.Size([4, 9, 768])
    Run on cpu
    tok_emb torch.Size([4, 10, 768])
    x after position_embedding: torch.Size([4, 10, 768])
    enc_out: torch.Size([4, 10, 768])
    x after decoder: torch.Size([4, 10, 768])
    Generate result:  torch.Size([4, 11])
    """

    print("result: \n", result)
    """
    输出：
    result:
    tensor([[ 1,  8,  4,  3,  8,  7,  1,  7, 14, 25, 14],
            [ 3,  4,  3,  6,  2,  8,  3,  4, 14,  3, 22],
            [ 5,  3,  1,  4,  5,  5,  5,  8, 23,  2, 28],
            [ 5,  4,  6,  8,  9,  3,  1,  4, 19, 31, 25]])
    """

def get_model_gpu():
    device = "cuda"
    model_config = TransformerConfig(
        block_size = 128,
        vocab_size = 32,
        n_layer = 4,
        n_head = 4,
        n_embd = 768,
        dropout = 0.0,
        bias = True
    )

    # print("model_config: \n", model_config)
    model = Transformer(model_config)
    model.to(device)
    # print("model:\n", model)

    # 前向传播
    input = torch.randint(1, 10, (4, 8)).to(device)
    logits, _ = model(input)
    print("input: ", input.size())
    print("logits: ", logits.size())
    """
    输出：
    number of parameters: 63.14M parameters
    Run on cuda:0
    tok_emb torch.Size([4, 8, 768])
    x after position_embedding: torch.Size([4, 8, 768])
    enc_out: torch.Size([4, 8, 768])
    x after decoder: torch.Size([4, 8, 768])
    input:  torch.Size([4, 8])
    logits:  torch.Size([4, 1, 32])
    """

    # 推理
    result = model.generate(input, 3)
    print("Generate result: ", result.size())
    """
    输出：
    Run on cuda:0
    tok_emb torch.Size([4, 8, 768])
    x after position_embedding: torch.Size([4, 8, 768])
    enc_out: torch.Size([4, 8, 768])
    x after decoder: torch.Size([4, 8, 768])
    Run on cuda:0
    tok_emb torch.Size([4, 9, 768])
    x after position_embedding: torch.Size([4, 9, 768])
    enc_out: torch.Size([4, 9, 768])
    x after decoder: torch.Size([4, 9, 768])
    Run on cuda:0
    tok_emb torch.Size([4, 10, 768])
    x after position_embedding: torch.Size([4, 10, 768])
    enc_out: torch.Size([4, 10, 768])
    x after decoder: torch.Size([4, 10, 768])
    Generate result:  torch.Size([4, 11])
    """

    print("result: \n", result)
    """
    输出：
    result:
    tensor([[ 6,  8,  5,  6,  1,  8,  1,  3,  6, 25,  3],
            [ 5,  1,  8,  9,  1,  8,  4,  2, 14, 31, 22],
            [ 7,  5,  2,  2,  6,  4,  7,  1,  1, 18, 23],
            [ 1,  6,  1,  6,  5,  3,  9,  2,  4, 19,  2]], device='cuda:0')
    """




if __name__ == "__main__":
    get_model_cpu()
    get_model_gpu()
