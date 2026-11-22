import os
import torch
import pandas as pd
import tiktoken
from tqdm import tqdm
import requests

MAX_STORIES = 10000 
URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/train-00000-of-00004-2d5a1467fff1081b.parquet"
LOCAL_FILE = "tiny_stories.parquet"
EXPECTED_SIZE = 249000000

def download_file(url, filename):
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    if os.path.exists(filename) and os.path.getsize(filename) == total_size:
        print("File already exists and looks complete!")
        return
    with open(filename, 'wb') as f, tqdm(desc=filename, total=total_size, unit='iB', unit_scale=True) as bar:
        for data in response.iter_content(chunk_size=1024*1024):
            size = f.write(data)
            bar.update(size)

def main():
    if not os.path.exists(LOCAL_FILE) or os.path.getsize(LOCAL_FILE) < EXPECTED_SIZE:
        download_file(URL, LOCAL_FILE)
    else:
        print("1. File already downloaded and looks good!")

    print("2. Loading data into memory...")
    df = pd.read_parquet(LOCAL_FILE)
    texts = df["text"].tolist()[:MAX_STORIES]
    print(f"   -> Successfully loaded {len(texts)} stories.")

    # --- THE FIX: Use tiktoken instead of our slow Python BPE ---
    print("3. Loading production BPE Tokenizer (tiktoken)...")
    # 'gpt2' uses a 50,257 vocab. We will stick with it for maximum speed.
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    print(f"   -> Tokenizer ready! Vocab size: {vocab_size}")

    # 4. Tokenize
    print(f"4. Tokenizing {len(texts)} stories...")
    all_tokens = []
    for text in tqdm(texts):
        # Add <|endoftext|> token ID (50256) to separate stories
        tokens = tokenizer.encode(text) + [50256]
        all_tokens.extend(tokens)

    # 5. Save Tensor
    print(f"\n5. Total tokens: {len(all_tokens):,}")
    # tiktoken vocab is 50k, so we need int32, not uint16
    tensor_data = torch.tensor(all_tokens, dtype=torch.int32)
    torch.save(tensor_data, "train_data.pt")
    
    # Save vocab size for the model script
    with open("vocab_size.txt", "w") as f:
        f.write(str(vocab_size))
        
    print("✅ Data saved to train_data.pt!")

if __name__ == "__main__":
    main()