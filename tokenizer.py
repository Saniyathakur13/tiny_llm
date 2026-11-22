import json

# Read dataset
text = open("dataset.txt", "r").read()
words = text.split()

# Create vocabulary
vocab = sorted(set(words))

word_to_id = {word: i for i, word in enumerate(vocab)}
id_to_word = {i: word for word, i in word_to_id.items()}

# Save vocabulary
with open("vocab.json", "w") as file:
    json.dump(word_to_id, file, indent=4)

print(f"Vocab size is now: {len(vocab)}")
print("Vocabulary saved to vocab.json")