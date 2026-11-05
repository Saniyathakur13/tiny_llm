import torch
import torch.nn as nn

# Settings
vocab_size = 256
embedding_size = 8
block_size = 4

# -------------------------
# 1. Create embeddings
# -------------------------

token_embedding = nn.Embedding(
    vocab_size,
    embedding_size
)

position_embedding = nn.Embedding(
    block_size,
    embedding_size
)

# Example input
tokens = torch.tensor([1, 13, 4, 3])

# Token embeddings
token_vectors = token_embedding(tokens)

# Position embeddings
positions = torch.arange(block_size)
position_vectors = position_embedding(positions)

# Final input
x = token_vectors + position_vectors

print("Input shape:")
print(x.shape)


# -------------------------
# 2. Create Query, Key, Value
# -------------------------

query_layer = nn.Linear(embedding_size, embedding_size)
key_layer = nn.Linear(embedding_size, embedding_size)
value_layer = nn.Linear(embedding_size, embedding_size)

Q = query_layer(x)
K = key_layer(x)
V = value_layer(x)

print("\nQuery shape:")
print(Q.shape)

print("\nKey shape:")
print(K.shape)

print("\nValue shape:")
print(V.shape)
# -------------------------
# 3. Calculate attention scores
# -------------------------

import math

scores = (Q @ K.T) / math.sqrt(embedding_size)

mask = torch.tril(torch.ones(block_size, block_size))

scores = scores.masked_fill(
    mask == 0,
    float("-inf")
)

attention_weights = torch.softmax(
    scores,
    dim=-1
)

output = attention_weights @ V
print("\nCausal mask:")
print(mask)
print("\nAttention scores:")
print(scores)

print("\nAttention score shape:")
print(scores.shape)
# -------------------------
# 4. Convert scores to weights
# -------------------------

attention_weights = torch.softmax(
    scores,
    dim=-1
)

print("\nAttention weights:")
print(attention_weights)

print("\nRow sums:")
print(attention_weights.sum(dim=-1))
# -------------------------
# 5. Weighted values
# -------------------------

output = attention_weights @ V

print("\nAttention output:")
print(output)

print("\nOutput shape:")
print(output.shape)