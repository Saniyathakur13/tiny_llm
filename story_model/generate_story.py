import torch
import tiktoken
from story_model import TinyStoriesModel, block_size

# Set the device
device = 'cpu'

# 1. Load Tokenizer
print("Loading Tokenizer...")
tokenizer = tiktoken.get_encoding("gpt2")

# 2. Load Model
print("Loading Model weights...")
model = TinyStoriesModel().to(device)
# Load the checkpoint we just trained
model.load_state_dict(torch.load("story_model.pth", map_location=device))
model.eval() # Set to evaluation mode (no dropout)

# 3. Generation Function
@torch.no_grad()
def generate(prompt, max_new_tokens=100, temperature=0.8):
    # Encode the prompt
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0) # Add batch dimension
    
    for _ in range(max_new_tokens):
        # Crop the context to the last block_size tokens
        tokens_cond = tokens[:, -block_size:]
        
        # Get predictions
        logits, _ = model(tokens_cond)
        
        # Focus only on the last time step (the prediction for the next token)
        logits = logits[:, -1, :] 
        
        # Apply temperature
        logits = logits / temperature
        
        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Sample from the distribution
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append the new token to the sequence
        tokens = torch.cat((tokens, next_token), dim=1)
        
        # Stop if we hit the end of text token (50256)
        if next_token.item() == 50256:
            break
            
    # Decode back to text
    return tokenizer.decode(tokens[0].tolist())

# 4. Test it out!
if __name__ == "__main__":
    prompt = "Once upon a time"
    print(f"\nPrompt: {prompt}")
    print("-" * 40)
    story = generate(prompt, max_new_tokens=150, temperature=0.8)
    print(story)
    print("-" * 40)