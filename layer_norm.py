import torch
import torch.nn as nn

x = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [10.0, 20.0, 30.0, 40.0]
])

layer_norm = nn.LayerNorm(4)

output = layer_norm(x)

print("Before LayerNorm:")
print(x)

print("\nAfter LayerNorm:")
print(output)

print("\nShape:")
print(output.shape)