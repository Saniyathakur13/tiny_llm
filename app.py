import streamlit as st
import torch
import json

from model import TinyLanguageModel


# -----------------------------
# Load vocabulary
# -----------------------------

with open("vocab.json", "r") as file:
    word_to_id = json.load(file)

word_to_id = {
    word: int(token_id)
    for word, token_id in word_to_id.items()
}

id_to_word = {
    token_id: word
    for word, token_id in word_to_id.items()
}


# -----------------------------
# Load trained model
# -----------------------------

model = TinyLanguageModel()

model.load_state_dict(
    torch.load(
        "tiny_llm.pth",
        map_location="cpu"
    )
)

model.eval()


# -----------------------------
# Text generation function
# -----------------------------

def generate_text(text, temperature, num_tokens):

    words = text.split()

    tokens = []

    for word in words:

        if word not in word_to_id:
            return "Unknown word: " + word

        tokens.append(
            word_to_id[word]
        )

    tokens = torch.tensor([tokens])

    for i in range(num_tokens):

        input_tokens = tokens[:, -4:]

        with torch.no_grad():

            logits = model(input_tokens)

        last_logits = logits[:, -1, :]

        probabilities = torch.softmax(
            last_logits / temperature,
            dim=-1
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        tokens = torch.cat(
            [
                tokens,
                next_token
            ],
            dim=1
        )

    generated_words = []

    for token_id in tokens[0]:

        word = id_to_word[
            token_id.item()
        ]

        generated_words.append(word)

    return " ".join(generated_words)


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("Tiny Transformer Language Model")

st.write(
    "My small language model built with PyTorch"
)

text = st.text_input(
    "Enter starting text:",
    value="Python is"
)

temperature = st.slider(
    "Temperature",
    min_value=0.1,
    max_value=2.0,
    value=0.7,
    step=0.1
)

num_tokens = st.slider(
    "Number of new tokens",
    min_value=1,
    max_value=20,
    value=5,
    step=1
)

if st.button("Generate"):

    result = generate_text(
        text,
        temperature,
        num_tokens
    )

    st.subheader("Generated text")

    st.write(result)