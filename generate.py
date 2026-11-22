import json
import torch
from model import TinyLanguageModel

model = TinyLanguageModel()
model.load_state_dict(torch.load("tiny_llm.pth"))
model.eval()

with open("vocab.json", "r") as file:
    word_to_id = json.load(file)

word_to_id = {word: int(token_id) for word, token_id in word_to_id.items()}
id_to_word = {token_id: word for word, token_id in word_to_id.items()}

text = input("Enter starting text: ")
words = text.split()
tokens = [word_to_id[word] for word in words if word in word_to_id]

if len(tokens) == 0:
    print("None of those words are in the vocab!")
    exit()

tokens = torch.tensor([tokens])

max_tokens = 20 # It will stop early if it hits <EOS>
eos_token_id = word_to_id.get("<EOS>", -1)

print("\nGenerating...")
for i in range(max_tokens):
    input_tokens = tokens[:, -4:]
    
    with torch.no_grad():
        logits = model(input_tokens)
    
    last_logits = logits[:, -1, :]
    
    # Temperature actually applied now
    temperature = 0.7 
    probabilities = torch.softmax(last_logits / temperature, dim=-1)
    
    next_token = torch.multinomial(probabilities, num_samples=1)
    tokens = torch.cat([tokens, next_token], dim=1)

    # STOP if we hit the EOS token!
    if next_token.item() == eos_token_id:
        break

generated_words = []
for token_id in tokens[0]:
    word = id_to_word[token_id.item()]
    if word != "<EOS>": # Don't print the <EOS> token to the user
        generated_words.append(word)

print("\nGenerated text:")
print(" ".join(generated_words))