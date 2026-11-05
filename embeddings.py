import torch
import torch.nn as nn

# Our vocabulary
vocab_size = 256

# Size of each embedding vector
embedding_size = 8

# Create embedding layer
embedding = nn.Embedding(vocab_size, embedding_size)

# Example token IDs
tokens = torch.tensor([1, 13, 4])

# Convert token IDs into vectors
vectors = embedding(tokens)

print("Token IDs:")
print(tokens)

print("\nEmbeddings:")
print(vectors)

print("\nShape:")
print(vectors.shape)