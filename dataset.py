import json
import torch

# Load vocabulary
with open("vocab.json", "r") as file:
    word_to_id = json.load(file)

# Convert token IDs from strings to integers
word_to_id = {
    word: int(token_id)
    for word, token_id in word_to_id.items()
}

# Read the new dataset
text = open("dataset.txt", "r").read()

# Convert every word into its token ID
tokens = [
    word_to_id[word]
    for word in text.split()
]

tokens = torch.tensor(tokens)

print("Number of tokens:")
print(len(tokens))

print("\nFirst 20 tokens:")
print(tokens[:20])

print("\nTensor shape:")
print(tokens.shape)