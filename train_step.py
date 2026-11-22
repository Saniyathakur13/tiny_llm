import torch
import torch.nn as nn
import json
from model import TinyLanguageModel
from dataset import tokens

block_size = 4

# Load vocab size
with open("vocab.json", "r") as f:
    vocab_size = len(json.load(f))

# Create input and target sequences
x = []
y = []

for i in range(len(tokens) - block_size):
    input_chunk = tokens[i:i + block_size]
    target_chunk = tokens[i + 1:i + block_size + 1]
    x.append(input_chunk)
    y.append(target_chunk)

x = torch.stack(x)
y = torch.stack(y)

model = TinyLanguageModel()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Starting training...")
for step in range(2000):
    logits = model(x)
    
    # We use vocab_size here instead of hardcoded 256
    loss = loss_function(logits.view(-1, vocab_size), y.view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print("Step:", step, "Loss:", loss.item())

torch.save(model.state_dict(), "tiny_llm.pth")
print("Model saved!")