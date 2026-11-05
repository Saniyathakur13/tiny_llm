import torch
import torch.nn as nn

# Our embedding size
embedding_size = 8

# Feed Forward Network
ffn = nn.Sequential(
    nn.Linear(embedding_size, 32),
    nn.ReLU(),
    nn.Linear(32, embedding_size)
)

# Example input
x = torch.randn(4, embedding_size)

print("Input:")
print(x)

print("\nInput shape:")
print(x.shape)

# Pass through FFN
output = ffn(x)

print("\nFFN output:")
print(output)

print("\nOutput shape:")
print(output.shape)