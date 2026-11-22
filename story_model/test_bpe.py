from bpe_tokenizer import BPETokenizer

# 1. Sample text (a tiny story)
text = "Once upon a time, there was a little girl named Lily. She loved to play in the garden."

# 2. Train the tokenizer (tiny vocab for a quick test)
tokenizer = BPETokenizer(vocab_size=300)
print("Training Tokenizer...")
tokenizer.train(text, verbose=True)

# 3. Test Encode/Decode
print("\n--- Testing ---")
encoded = tokenizer.encode("Lily loved the garden.")
print(f"Original: 'Lily loved the garden.'")
print(f"Encoded IDs: {encoded}")
print(f"Decoded: '{tokenizer.decode(encoded)}'")