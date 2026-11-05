import torch

# Original input
x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

# Imagine this came from attention
attention_output = torch.tensor([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6]
])

# Residual connection
output = x + attention_output

print("Original input:")
print(x)

print("\nAttention output:")
print(attention_output)

print("\nAfter residual connection:")
print(output)