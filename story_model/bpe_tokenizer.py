import json
from collections import Counter

class BPETokenizer:
    def __init__(self, vocab_size=4096):
        self.target_vocab_size = vocab_size
        self.merges = {} # Maps a pair (a, b) -> new_token_id
        self.vocab = {i: bytes([i]) for i in range(256)} # Start with 256 raw bytes
        
    def get_stats(self, ids):
        """Counts how often every pair of tokens appears next to each other."""
        counts = Counter()
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts

    def merge(self, ids, pair, new_id):
        """Replaces every instance of 'pair' in the sequence with 'new_id'."""
        new_ids = []
        i = 0
        while i < len(ids):
            # If we find the pair, replace it with the new token
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(new_id)
                i += 2 # Skip the next token since we merged it
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text, verbose=True):
        """Learns the merges from raw text."""
        # 1. Convert text to raw bytes (UTF-8)
        ids = list(text.encode("utf-8"))
        
        # 2. Iteratively merge the most frequent pairs
        num_merges = self.target_vocab_size - 256
        for i in range(num_merges):
            counts = self.get_stats(ids)
            if not counts:
                break
                
            # Find the absolute most frequent pair
            most_common_pair = max(counts, key=counts.get)
            
            # The new token ID will be 256, 257, 258, etc.
            new_id = 256 + i
            
            # Apply the merge
            ids = self.merge(ids, most_common_pair, new_id)
            
            # Save the rule
            self.merges[most_common_pair] = new_id
            self.vocab[new_id] = self.vocab[most_common_pair[0]] + self.vocab[most_common_pair[1]]
            
            if verbose and i % 500 == 0:
                print(f"Merge {i+1}/{num_merges}: {most_common_pair} -> {new_id} (freq: {counts[most_common_pair]})")

    def encode(self, text):
        """Converts raw text into a list of token IDs using learned merges."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            counts = self.get_stats(ids)
            # Find the pair that we have a merge rule for, with the LOWEST new_id
            # (This ensures we apply the merges in the exact order we learned them)
            pair_to_merge = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            
            # If no pair can be merged anymore, break
            if pair_to_merge not in self.merges:
                break
                
            new_id = self.merges[pair_to_merge]
            ids = self.merge(ids, pair_to_merge, new_id)
        return ids

    def decode(self, ids):
        """Converts token IDs back to raw text."""
        # Join the bytes together and decode as UTF-8
        part_bytes = b"".join(self.vocab[idx] for idx in ids)
        return part_bytes.decode("utf-8", errors="replace")

    def save(self, path):
        """Save the merges and vocab to a JSON file."""
        # Convert tuple keys to strings for JSON serialization
        merges_str_keys = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        with open(path, "w") as f:
            json.dump({"merges": merges_str_keys, "vocab_size": self.target_vocab_size}, f)

    @classmethod
    def load(cls, path):
        """Load tokenizer from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        tokenizer = cls(vocab_size=data["vocab_size"])
        tokenizer.merges = {tuple(map(int, k.split(","))): v for k, v in data["merges"].items()}
        # Rebuild vocab
        for pair, idx in tokenizer.merges.items():
            tokenizer.vocab[idx] = tokenizer.vocab[pair[0]] + tokenizer.vocab[pair[1]]
        return tokenizer