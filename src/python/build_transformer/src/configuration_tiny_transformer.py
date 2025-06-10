from dataclasses import dataclass

@dataclass
class TransformerConfig:
    block_size: int = 1024,
    vocab_size: int = 50304,
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
