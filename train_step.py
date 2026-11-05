import torch
import torch.nn as nn

from model import TinyLanguageModel

block_size = 4

# Load the complete dataset
from dataset import tokens

# Create input and target sequences
x = []
y = []

for i in range(len(tokens) - block_size):

    input_chunk = tokens[i:i + block_size]

    target_chunk = tokens[
        i + 1:i + block_size + 1
    ]

    x.append(input_chunk)
    y.append(target_chunk)

x = torch.stack(x)
y = torch.stack(y)

print("Input shape:")
print(x.shape)

print("\nTarget shape:")
print(y.shape)


# Create model
model = TinyLanguageModel()

# Loss function
loss_function = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# Training
for step in range(2000):

    logits = model(x)

    loss = loss_function(
        logits.view(-1, 256),
        y.view(-1)
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % 200 == 0:

        print(
            "Step:",
            step,
            "Loss:",
            loss.item()
        )


# Save trained model
torch.save(
    model.state_dict(),
    "tiny_llm.pth"
)

print("\nModel saved!")