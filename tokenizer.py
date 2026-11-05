import json


# -------------------------
# Read dataset
# -------------------------

text = open("dataset.txt", "r").read()

words = text.split()


# -------------------------
# Create vocabulary
# -------------------------

vocab = sorted(set(words))

word_to_id = {
    word: i
    for i, word in enumerate(vocab)
}

id_to_word = {
    i: word
    for word, i in word_to_id.items()
}


# -------------------------
# Save vocabulary
# -------------------------

with open("vocab.json", "w") as file:

    json.dump(
        word_to_id,
        file,
        indent=4
    )


print("Vocabulary:")
print(word_to_id)

print("\nVocabulary saved to vocab.json")


# -------------------------
# Test tokenizer
# -------------------------

sentence = "Python is useful for machine learning."

tokens = [
    word_to_id[word]
    for word in sentence.split()
]

print("\nSentence:")
print(sentence)

print("\nTokens:")
print(tokens)