import torch
import torch.nn as nn
import json
from transformer_block import TransformerBlock

# Dynamically load vocab size from the file we just created
with open("vocab.json", "r") as f:
    vocab_size = len(json.load(f))

embedding_size = 8
block_size = 4

class TinyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(block_size, embedding_size)
        self.transformer = TransformerBlock()
        self.output_layer = nn.Linear(embedding_size, vocab_size)

    def forward(self, tokens):
        B, T = tokens.shape
        token_vectors = self.token_embedding(tokens)
        positions = torch.arange(T, device=tokens.device)
        position_vectors = self.position_embedding(positions)
        x = token_vectors + position_vectors
        x = self.transformer(x)
        logits = self.output_layer(x)
        return logits