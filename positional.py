import torch
import torch.nn as nn

vocab_size = 256
embedding_size = 8
block_size = 4

# Token embeddings
token_embedding = nn.Embedding(
    vocab_size,
    embedding_size
)

# Position embeddings
position_embedding = nn.Embedding(
    block_size,
    embedding_size
)

# Example input
tokens = torch.tensor([1, 13, 4, 2])

# Token vectors
token_vectors = token_embedding(tokens)

# Position numbers
positions = torch.arange(0, block_size)

# Position vectors
position_vectors = position_embedding(positions)

# Combine them
x = token_vectors + position_vectors

print("Token vectors:")
print(token_vectors)

print("\nPosition vectors:")
print(position_vectors)

print("\nFinal input:")
print(x)

print("\nShape:")
print(x.shape)