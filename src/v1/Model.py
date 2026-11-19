from torch import nn
from Transformers import Transformer, RMSNorm

# Define the language model structure
class LanguageModel(nn.Module):
    def __init__(self, vocabulary_size, dim, num_heads, num_blocks, hidden):
        super().__init__()

        self.token_embedding = nn.Embedding(vocabulary_size, dim)

        self.blocks = nn.ModuleList([
            Transformer(dim, hidden, num_heads)
            for _ in range(num_blocks)
        ])

        self.norm = RMSNorm(dim)

        self.out = nn.Linear(dim, vocabulary_size, bias=False)
        self.out.weight = self.token_embedding.weight

    def forward(self, x):
        x = self.token_embedding(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return self.out(x)

# Example Model V1
from Tokenizer import Tokenizer
tokenizer = Tokenizer("../../example_data/encoder.json", "example_data/vocab.bpe")

V1_Base = LanguageModel(tokenizer.vocab_size, 1024, 16, 4096, 24)