import torch
import torch.nn as nn


# --------------------------------
# Example model predictions
# --------------------------------

logits = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [4.0, 3.0, 2.0, 1.0]
])


# --------------------------------
# Correct answers
# --------------------------------

targets = torch.tensor([
    3,
    0
])


# --------------------------------
# Cross Entropy Loss
# --------------------------------

loss_function = nn.CrossEntropyLoss()

loss = loss_function(
    logits,
    targets
)


print("Logits:")
print(logits)

print("\nTargets:")
print(targets)

print("\nLoss:")
print(loss)