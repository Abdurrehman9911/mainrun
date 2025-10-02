import torch
import json
from pathlib import Path
from tokenizers import Tokenizer

# We need to redefine the model architecture classes here
# Or you could import them from train.py
from train import GPT, GPTConfig, Hyperparameters

def generate_text(model, tokenizer, device, max_new_tokens=50):
    """
    Generates text from the model.
    """
    model.eval()
    
    # The <eos> token was used to separate titles during training.
    # We use its ID as the starting prompt for the model to generate a new title.
    start_token_id = tokenizer.token_to_id("<eos>")
    
    # The context is a tensor of shape (B, T) with B=1 (batch size)
    context = torch.tensor([[start_token_id]], dtype=torch.long, device=device)
    
    print("Generating new title...")
    
    # Generate tokens autoregressively
    generated_ids = model.generate(context, max_new_tokens=max_new_tokens, eos_token_id=start_token_id)
    
    # Decode the generated IDs into a string
    generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
    
    print("-" * 30)
    print(generated_text)
    print("-" * 30)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Artifacts ---
    artifact_path = Path("./model_artifacts")
    if not artifact_path.exists():
        print("Error: `model_artifacts` directory not found. Please run train.py first.")
        return

    # Load the tokenizer
    tokenizer = Tokenizer.from_file(str(artifact_path / "tokenizer.json"))

    # --- Recreate Model Architecture ---
    # We use the hyperparameters to configure the model architecture correctly.
    args = Hyperparameters()
    cfg = GPTConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)
    
    # --- Load Model Weights ---
    try:
        model.load_state_dict(torch.load(artifact_path / "model_weights.pth", map_location=device))
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print("Error: model_weights.pth not found. Please ensure training is complete.")
        return
    
    # Add a simple generate method to the GPT class for convenience
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=20, eos_token_id=None):
        for _ in range(max_new_tokens):
            # Crop context if it's longer than block_size
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            # Forward pass
            logits, _ = self(idx_cond)
            # Get the logits for the last time step
            logits = logits[:, -1, :]
            
            # Optionally apply top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply temperature
            logits = logits / temperature
            # Get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # Sample the next token
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append to the sequence
            idx = torch.cat((idx, idx_next), dim=1)

            # Stop if we generate the EOS token
            if eos_token_id is not None and idx_next.item() == eos_token_id:
                break
        return idx

    # Monkey-patch the generate method onto our loaded model instance
    GPT.generate = generate
    
    generate_text(model, tokenizer, device)


if __name__ == "__main__":
    main()