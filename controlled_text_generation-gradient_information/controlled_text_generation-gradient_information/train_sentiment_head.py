#!/usr/bin/env python3
"""
train_sentiment_head.py

Trains a lightweight sentiment classifier that lives ON TOP of the frozen GPT-2
you already use as the energy function.

Why not just reuse MuCoLa's RoBERTa classifier?
  MuCoLa feeds the generator's embeddings straight into the classifier via
  `inputs_embeds` (see mucoco/losses/classification.py). That only works because
  they use a *custom* RoBERTa retrofitted to share GPT-2's vocabulary and
  embedding table (RobertaCustomForSequenceClassification). Off-the-shelf RoBERTa
  has a different tokenizer and embedding space, so GPT-2 embeddings are
  meaningless to it and the gradient you would get back is garbage.

  Training a head on the frozen GPT-2's own hidden states sidesteps that entirely:
  the embedding space matches by construction, gradients flow natively back to the
  input embeddings, and the constraint is a genuine differentiable function of the
  same variable the sampler optimizes. This is the faithful way to ask our question,
  which is not "can we clone MuCoLa" but "does a *classifier* gradient carry the
  directional signal that the *LM likelihood* gradient does not?"

The head is a mean-pooled linear probe over the frozen GPT-2's last hidden state.
The backbone stays frozen, so the energy landscape of the LM is untouched; we are
only adding a differentiable readout.

Data: SST-2 (same dataset MuCoLa's sentiment classifier uses).

Usage:
  python train_sentiment_head.py \
      --base /path/to/gpt2_large_sft_output \
      --out sentiment_head_gpt2large.pt --epochs 2
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
        # hidden: (B, T, H). Mean-pool over real tokens.
        if attn_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            m = attn_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return self.proj(pooled)   # logits (B, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="the SAME frozen GPT-2 used as the energy")
    ap.add_argument("--out", default="sentiment_head.pt")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load_tokenizer_and_model(args.base, dtype=torch.float32)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    from datasets import load_dataset
    ds = load_dataset("glue", "sst2")

    def collate(batch):
        texts = [b["sentence"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch])
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
        return enc.input_ids, enc.attention_mask, labels

    train = DataLoader(ds["train"], batch_size=args.bs, shuffle=True, collate_fn=collate)
    val = DataLoader(ds["validation"], batch_size=args.bs, collate_fn=collate)

    hidden_size = model.config.hidden_size
    head = SentimentHead(hidden_size).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        head.train()
        for i, (ids, mask, y) in enumerate(train):
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            with torch.no_grad():
                h = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
            logits = head(h, mask)
            loss = lossf(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            if i % 200 == 0:
                print(f"  ep{ep} step{i} loss={loss.item():.4f}", flush=True)

        head.eval(); correct = tot = 0
        with torch.no_grad():
            for ids, mask, y in val:
                ids, mask, y = ids.to(device), mask.to(device), y.to(device)
                h = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
                correct += (head(h, mask).argmax(-1) == y).sum().item(); tot += len(y)
        print(f"epoch {ep}: SST-2 val accuracy = {100*correct/tot:.2f}%", flush=True)

    torch.save({"state_dict": head.state_dict(), "hidden_size": hidden_size}, args.out)
    print(f"saved head -> {args.out}")
    print("If val accuracy is below ~85%, the probe is too weak to be a meaningful")
    print("constraint and the ablation will be uninformative. Train longer or unfreeze more.")


if __name__ == "__main__":
    main()
