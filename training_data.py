import torch


# --------------------------------
# Complete token sequence
# --------------------------------

tokens = torch.tensor([
    1, 13, 4, 3, 10, 6,
    3, 10, 15, 1, 13, 14,
    12, 2, 11, 10, 9, 1,
    13, 5, 8, 0, 7, 10,
    15, 3, 10, 16, 2, 11,
    10, 16
])


# --------------------------------
# Create input and target
# --------------------------------

x = tokens[:-1]

y = tokens[1:]


print("Input:")
print(x)

print("\nTarget:")
print(y)

print("\nInput shape:")
print(x.shape)

print("\nTarget shape:")
print(y.shape)