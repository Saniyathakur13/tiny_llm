import torch
torch.set_num_threads(4) # Force use of all 4 Ryzen cores
import time
import os

# Import the model we just created
from story_model import TinyStoriesModel, block_size

# --- TRAINING SETTINGS ---
batch_size = 16        # How many chunks to process at once
learning_rate = 1e-3   # How fast the model learns
max_iters = 4000       # Total training steps (this will take a few hours on your CPU)
eval_interval = 500    # How often to save a checkpoint
eval_iters = 50        # How many batches to average for validation

device = 'cpu' # Your Ryzen 5 CPU
print(f"Using device: {device}")

# 1. Load the data we prepared
print("Loading data...")
data = torch.load("train_data.pt", weights_only=True)
print(f"Total tokens available: {len(data):,}")

# Split into train and validation (90% train, 10% validation)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    """Gets a random batch of input (x) and target (y) sequences"""
    data_split = train_data if split == 'train' else val_data
    # Pick random starting points
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    # Stack the sequences
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    # BUG FIX: PyTorch requires targets to be 64-bit integers (Long)
    return x.to(device), y.to(device).long()

@torch.no_grad()
def estimate_loss(model):
    """Calculates average loss on train and val sets"""
    out = {}
    model.eval() # Set model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Set model back to training mode
    return out

# 2. Initialize Model and Optimizer
print("Initializing 13M parameter model...")
model = TinyStoriesModel().to(device)
# AdamW is the standard optimizer for Transformers
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# --- RESUME FROM CHECKPOINT ---
if os.path.exists("story_model.pth"):
    print("Found existing checkpoint! Loading weights...")
    model.load_state_dict(torch.load("story_model.pth", map_location=device))
    print("Resuming training from step 0 (but with pre-trained weights)...")

# 3. Training Loop
print("Starting training...")
start_time = time.time()

for iter in range(max_iters):
    
    # Evaluate loss every `eval_interval` steps
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        time_elapsed = (time.time() - start_time) / 60
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} | Time: {time_elapsed:.1f} min")
        
        # Save the model checkpoint
        torch.save(model.state_dict(), "story_model.pth")
        print("  -> Saved checkpoint to story_model.pth")

    # Get a batch of data
    xb, yb = get_batch('train')

    # Forward pass
    logits, loss = model(xb, yb)
    
    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Training complete!")
torch.save(model.state_dict(), "story_model.pth")
print("Final model saved to story_model.pth")