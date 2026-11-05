import torch
import torch.nn as nn
import math

# -------------------------
# Settings
# -------------------------

vocab_size = 256
embedding_size = 8
block_size = 4

num_heads = 2
head_size = embedding_size // num_heads


# -------------------------
# Token + Position Embeddings
# -------------------------

token_embedding = nn.Embedding(
    vocab_size,
    embedding_size
)

position_embedding = nn.Embedding(
    block_size,
    embedding_size
)

tokens = torch.tensor([1, 13, 4, 3])

positions = torch.arange(block_size)

x = (
    token_embedding(tokens)
    + position_embedding(positions)
)

print("Input shape:")
print(x.shape)


# -------------------------
# Create Q, K, V
# -------------------------

query_layer = nn.Linear(
    embedding_size,
    embedding_size
)

key_layer = nn.Linear(
    embedding_size,
    embedding_size
)

value_layer = nn.Linear(
    embedding_size,
    embedding_size
)

Q = query_layer(x)
K = key_layer(x)
V = value_layer(x)

print("\nQ shape:")
print(Q.shape)

print("\nK shape:")
print(K.shape)

print("\nV shape:")
print(V.shape)


# -------------------------
# Split into heads
# -------------------------

Q = Q.view(
    block_size,
    num_heads,
    head_size
)

K = K.view(
    block_size,
    num_heads,
    head_size
)

V = V.view(
    block_size,
    num_heads,
    head_size
)

print("\nQ after splitting:")
print(Q.shape)

print("\nK after splitting:")
print(K.shape)

print("\nV after splitting:")
print(V.shape)
# -------------------------
# Transpose
# -------------------------

Q = Q.transpose(0, 1)
K = K.transpose(0, 1)
V = V.transpose(0, 1)

print("\nQ after transpose:")
print(Q.shape)
# -------------------------
# Attention scores
# -------------------------

scores = (Q @ K.transpose(-2, -1)) / math.sqrt(head_size)

print("\nAttention scores:")
print(scores.shape)
# -------------------------
# Causal mask
# -------------------------

mask = torch.tril(
    torch.ones(block_size, block_size)
)

scores = scores.masked_fill(
    mask == 0,
    float("-inf")
)
attention_weights = torch.softmax(
    scores,
    dim=-1
)
output = attention_weights @ V
print("\nAttention output shape:")
print(output.shape)
output = output.transpose(0, 1)

print("\nAfter transpose:")
print(output.shape)
output = output.contiguous().view(
    block_size,
    embedding_size
)

print("\nFinal multi-head output:")
print(output)

print("\nFinal shape:")
print(output.shape)