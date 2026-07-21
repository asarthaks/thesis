#!/usr/bin/env python
"""
train_noisy_classifier.py  -  Part G step 1.

A small sentiment classifier that reads a NOISY (absorbing-corrupted) token-id state,
the state SEDD occupies mid-denoising. It exists ONLY to guide SEDD generation; the
steering gain is scored by the SEPARATE concern-11 judge (the frozen-GPT-2 sentiment
head), never by this classifier. Strict role separation.

Why a new architecture: the concern-11 head is a probe on frozen GPT-2 hidden states,
and GPT-2 cannot ingest the SEDD MASK id (50257 is out of GPT-2's 50257-token range).
Masked input therefore forces its own embedding table over the SEDD vocab (50258, MASK
included). Otherwise it mirrors the existing head: mean-pool then Linear-Tanh-Linear,
with the noise level appended as an input feature (trivial).

Corruption is SEDD's forward absorbing process at a random noise level uniform over
the schedule: t ~ U(0,1); each token -> MASK independently with prob (1-eps)*t (the
loglinear move-chance); the noise feature is that move-chance. This is exactly the
family of states the guided sampler will query, so the classifier is trained on the
states it is evaluated on (the thesis's own principle, applied constructively).

Data: SST-2 (glue), the dataset behind the existing sentiment head.
Gate: held-out accuracy at LOW noise must be well above chance, else guidance is noise.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

MASK_TOKEN = 50257
VOCAB = 50258   # 50257 real + MASK
EPS = 1e-3      # loglinear noise eps (SEDD default)


class NoisyClassifier(nn.Module):
    def __init__(self, d_emb=256, hidden=256, vocab=VOCAB):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_emb)
        self.mlp = nn.Sequential(
            nn.Linear(d_emb + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, ids, noise_feat, attn_mask=None):
        # ids: (B,T) long incl MASK; noise_feat: (B,) move-chance in [0,1]
        e = self.emb(ids)                                # (B,T,d)
        if attn_mask is None:
            pooled = e.mean(dim=1)
        else:
            m = attn_mask.unsqueeze(-1).to(e.dtype)
            pooled = (e * m).sum(1) / m.sum(1).clamp(min=1e-6)
        x = torch.cat([pooled, noise_feat.unsqueeze(-1).to(pooled.dtype)], dim=-1)
        return self.mlp(x)                               # (B,2)


def corrupt_absorbing(ids, attn_mask, move_chance, generator=None):
    """Independent token -> MASK with prob move_chance (per-row), respecting pad mask."""
    B, T = ids.shape
    r = torch.rand(B, T, generator=generator, device=ids.device)
    mc = move_chance.view(B, 1)
    drop = (r < mc) & (attn_mask.bool())
    out = torch.where(drop, torch.full_like(ids, MASK_TOKEN), ids)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_revision/noisy_classifier.pt")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--d_emb", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    from datasets import load_dataset
    ds = load_dataset("glue", "sst2")

    def collate(batch):
        texts = [b["sentence"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch])
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_len)
        return enc.input_ids, enc.attention_mask, labels

    train = DataLoader(ds["train"], batch_size=args.bs, shuffle=True, collate_fn=collate)
    val = DataLoader(ds["validation"], batch_size=args.bs, collate_fn=collate)

    clf = NoisyClassifier(d_emb=args.d_emb).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    gen = torch.Generator(device=device).manual_seed(args.seed)
    curve = []
    for ep in range(args.epochs):
        clf.train()
        for i, (ids, mask, y) in enumerate(train):
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            t = torch.rand(ids.shape[0], generator=gen, device=device)
            mc = (1 - EPS) * t                              # move-chance ~ U(0,1)
            noisy = corrupt_absorbing(ids, mask, mc, generator=gen)
            logits = clf(noisy, mc, mask)
            loss = lossf(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            if i % 300 == 0:
                print(f"  ep{ep} step{i} loss={loss.item():.4f}", flush=True)

        # held-out accuracy at several fixed noise levels
        clf.eval()
        acc_by_noise = {}
        for mc_fixed in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
            correct = tot = 0
            with torch.no_grad():
                for ids, mask, y in val:
                    ids, mask, y = ids.to(device), mask.to(device), y.to(device)
                    mc = torch.full((ids.shape[0],), mc_fixed, device=device)
                    noisy = corrupt_absorbing(ids, mask, mc, generator=gen) if mc_fixed > 0 else ids
                    pred = clf(noisy, mc, mask).argmax(-1)
                    correct += (pred == y).sum().item(); tot += len(y)
            acc_by_noise[mc_fixed] = 100.0 * correct / tot
        curve.append({"epoch": ep, "acc_by_noise": acc_by_noise})
        print(f"epoch {ep}: " + ", ".join(f"mc{k}={v:.1f}%" for k, v in acc_by_noise.items()),
              flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"state_dict": clf.state_dict(), "d_emb": args.d_emb, "vocab": VOCAB,
                "eps": EPS, "curve": curve}, args.out)
    low = curve[-1]["acc_by_noise"][0.0]
    print(f"saved -> {args.out}; final low-noise (mc=0) val acc = {low:.2f}%")
    print("GATE: low-noise accuracy must be well above 50% chance or guidance is noise.")
    json.dump({"curve": curve, "final_low_noise_acc": low, "out": args.out},
              open(args.out.replace(".pt", "_train.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
