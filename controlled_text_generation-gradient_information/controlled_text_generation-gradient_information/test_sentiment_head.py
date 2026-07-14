#!/usr/bin/env python3
"""
test_sentiment_head.py

Loads a trained sentiment head and the underlying frozen GPT-2 backbone
to test sentiment predictions on a custom set of sentences.
"""

import torch
import torch.nn as nn
from core.prep import load_tokenizer_and_model

class SentimentHead(nn.Module):
    """Mean-pooled linear probe on frozen GPT-2 hidden states. 2 classes: 0=neg, 1=pos."""
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 2),
        )

    def forward(self, hidden, attn_mask=None):
        if attn_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            m = attn_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return self.proj(pooled)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Paths defined during your training run
    base_model_path = "/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/infill_subj_arithmetic/gpt2_large_sft_output"
    head_checkpoint_path = "/mount/arbeitsdaten/studenten1/singhsk/models/sentiment_constrained_ft_gpt2_large/sentiment_head.pt"

    # 1. Load tokenizer and backbone GPT-2
    print("Loading tokenizer and base model...")
    tok, model = load_tokenizer_and_model(base_model_path, dtype=torch.float32)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.to(device)
    model.eval()

    # 2. Load trained head weights
    print("Loading sentiment head weights...")
    checkpoint = torch.load(head_checkpoint_path, map_location=device)
    hidden_size = checkpoint["hidden_size"]
    
    head = SentimentHead(hidden_size).to(device)
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()

    # 3. Test sentences (Mix of clear and slight nuance)
    test_sentences = [
        "An absolute masterpiece, beautifully acted and directed.",
        "I loved every single second of this movie!",
        "A complete waste of time and money. Terrible plot.",
        "The film was okay, but the ending was extremely boring and predictable.",
        "This works surprisingly well and exceeded my expectations.",
        "It was an incredibly frustrating experience from start to finish."
    ]

    id_to_label = {0: "🔴 Negative", 1: "🟢 Positive"}

    print("\n" + "="*60)
    print("RUNNING INFERENCE TEST")
    print("="*60 + "\n")

    with torch.no_grad():
        for text in test_sentences:
            # Match the exact preprocessing used in training
            enc = tok([text], return_tensors="pt", padding=True, truncation=True, max_length=64)
            ids = enc.input_ids.to(device)
            mask = enc.attention_mask.to(device)

            # Get final layer hidden states from backbone
            outputs = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            h = outputs.hidden_states[-1]

            # Forward pass through classification probe
            logits = head(h, mask)
            probabilities = torch.softmax(logits, dim=-1)[0]
            prediction = logits.argmax(-1).item()

            print(f"Sentence: \"{text}\"")
            print(f"Predicted: {id_to_label[prediction]}")
            print(f"Confidence: [Neg: {probabilities[0].item():.4f} | Pos: {probabilities[1].item():.4f}]")
            print("-" * 60)

if __name__ == "__main__":
    main()