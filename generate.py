import json
import torch

from model import TinyLanguageModel


# -------------------------
# Load trained model
# -------------------------

model = TinyLanguageModel()

model.load_state_dict(
    torch.load("tiny_llm.pth")
)

model.eval()


# -------------------------
# Load vocabulary
# -------------------------

with open("vocab.json", "r") as file:

    word_to_id = json.load(file)


# JSON stores keys as strings,
# so convert IDs back to integers

word_to_id = {
    word: int(token_id)
    for word, token_id in word_to_id.items()
}


id_to_word = {
    token_id: word
    for word, token_id in word_to_id.items()
}


# -------------------------
# Get starting text
# -------------------------

text = input("Enter starting text: ")

words = text.split()

tokens = []

for word in words:

    if word in word_to_id:

        tokens.append(
            word_to_id[word]
        )

    else:

        print("Unknown word:", word)
        exit()


tokens = torch.tensor([
    tokens
])


# -------------------------
# Generate 5 tokens
# -------------------------

for i in range(5):

    # Keep only the latest 4 tokens

    input_tokens = tokens[:, -4:]

    with torch.no_grad():

        logits = model(input_tokens)

    # Get prediction for last position

    last_logits = logits[:, -1, :]

    # Highest-scoring token

    # Convert logits into probabilities
temperature = 2.0
probabilities = torch.softmax(
    last_logits,
    dim=-1
)

# Randomly choose a token
# using those probabilities

next_token = torch.multinomial(
    probabilities,
    num_samples=1
)

    # Add predicted token

tokens = torch.cat(
        [
            tokens,
            next_token
        ],
        dim=1
    )


# -------------------------
# Convert IDs to words
# -------------------------

generated_words = []

for token_id in tokens[0]:

    word = id_to_word[
        token_id.item()
    ]

    generated_words.append(word)


print("\nGenerated text:")
print(" ".join(generated_words))