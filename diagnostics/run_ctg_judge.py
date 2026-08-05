#!/usr/bin/env python3
"""
run_ctg_judge.py

The SCORING half of the generate/score split. Reads the per-sequence CSVs written by
`run_ctg.py` and scores the text with models that had no part in producing it, then
writes one JSON per scored run.

STRICT CLASSIFIER SEPARATION, which is the whole point of running this as a separate
process rather than inside the chain:

  STEERS   core/constraint.py:SentimentHead, a head on GPT-2-Large's own hidden states.
           It appears in the energy and in the proposal. It is never read here.
  SCORES   siebert/sentiment-roberta-large-english, an off-the-shelf sentiment
           classifier with no connection to this project, for ADHERENCE.
           Llama-3-8B, for external FLUENCY (per-token NLL of the generated span).

Neither judge is importable from any chain code, and the judges are loaded only in this
file, so they can never sit in VRAM next to a generator.

DIVERSITY (Study E) is computed here too, since it is a property of the accumulated
text rather than of any single chain: distinct-n, self-BLEU and an embedding-based
semantic spread, all per prompt and then averaged, so that variation ACROSS prompts is
not mistaken for variation within one.

    python diagnostics/run_ctg_judge.py --glob 'results/ctg/constrained/*.csv' \
        --out_dir results/ctg/judged --stage sentiment
    python diagnostics/run_ctg_judge.py --glob 'results/ctg/constrained/*.csv' \
        --out_dir results/ctg/judged --stage fluency --judge_path $LLAMA3
"""
import argparse
import csv
import glob
import itertools
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def span_of(row):
    """The GENERATED span only: the prompt is common to every arm, so scoring the whole
    string would dilute every metric by a constant prefix and make the arms look more
    alike than they are."""
    txt, start = row["text"], row.get("start_text", "")
    pr = row.get("prompt", "")
    return txt[len(pr):].strip() if pr and txt.startswith(pr) else txt


# ---------------------------------------------------------------- diversity
def distinct_n(texts, n):
    tot, seen = 0, set()
    for t in texts:
        w = t.split()
        gs = [tuple(w[i:i + n]) for i in range(max(len(w) - n + 1, 0))]
        tot += len(gs)
        seen.update(gs)
    return float(len(seen) / tot) if tot else float("nan")


def self_bleu(texts, n=4):
    """Mean over sequences of a uniform-weight BLEU-n against the other sequences of the
    same prompt, brevity penalty included. Lower means more diverse."""
    if len(texts) < 2:
        return float("nan")
    toks = [t.split() for t in texts]
    out = []
    for i, hyp in enumerate(toks):
        if not hyp:
            continue
        refs = [toks[j] for j in range(len(toks)) if j != i]
        logs = []
        for k in range(1, n + 1):
            hc = Counter(tuple(hyp[a:a + k]) for a in range(max(len(hyp) - k + 1, 0)))
            if not hc:
                logs.append(-9e9)
                continue
            mx = Counter()
            for r in refs:
                rc = Counter(tuple(r[a:a + k]) for a in range(max(len(r) - k + 1, 0)))
                for g, c in rc.items():
                    mx[g] = max(mx[g], c)
            clip = sum(min(c, mx[g]) for g, c in hc.items())
            logs.append(np.log(clip / sum(hc.values())) if clip else -9e9)
        rl = min((len(r) for r in refs), key=lambda L: (abs(L - len(hyp)), L))
        bp = 1.0 if len(hyp) > rl else np.exp(1 - rl / max(len(hyp), 1))
        out.append(bp * np.exp(np.mean(logs)))
    return float(np.mean(out)) if out else float("nan")


def semantic_spread(embs):
    """Mean pairwise cosine DISTANCE of the sequence embeddings. Higher is more spread."""
    if len(embs) < 2:
        return float("nan")
    x = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    sims = [float(x[i] @ x[j]) for i, j in itertools.combinations(range(len(x)), 2)]
    return float(1.0 - np.mean(sims))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", required=True, help="glob over run_ctg.py CSVs")
    p.add_argument("--out_dir", default="results/ctg/judged")
    p.add_argument("--stage", choices=["sentiment", "fluency"], required=True)
    p.add_argument("--sentiment_judge", default="siebert/sentiment-roberta-large-english")
    p.add_argument("--judge_path", default=None, help="fluency judge, e.g. Llama-3-8B")
    p.add_argument("--target_label", type=int, default=1, help="1 = POSITIVE")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--exclude_shards", action="store_true",
                   help="skip per-shard CSVs and score only merged runs. Diversity is a "
                        "property of the whole family: a shard holds too few sequences "
                        "per prompt for self-BLEU or semantic spread to be defined.")
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    paths = sorted(glob.glob(a.glob))
    if a.exclude_shards:
        paths = [q for q in paths if ".shard" not in os.path.basename(q)]
    if not paths:
        raise SystemExit(f"no CSVs matched {a.glob}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if a.stage == "sentiment":
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        jt = AutoTokenizer.from_pretrained(a.sentiment_judge)
        jm = AutoModelForSequenceClassification.from_pretrained(
            a.sentiment_judge, torch_dtype=torch.float32).to(device).eval()
        # embedding model for semantic spread: the judge's own encoder, mean-pooled.
        # It is a scorer, never a steerer, so reusing it here breaks no separation.
        print(f"[judge] sentiment judge {a.sentiment_judge}, {len(paths)} runs", flush=True)
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        if not a.judge_path:
            raise SystemExit("--stage fluency needs --judge_path")
        jt = AutoTokenizer.from_pretrained(a.judge_path)
        jm = AutoModelForCausalLM.from_pretrained(
            a.judge_path, torch_dtype=torch.bfloat16).to(device).eval()
        if jt.pad_token_id is None:
            jt.pad_token = jt.eos_token
        print(f"[judge] fluency judge {a.judge_path}, {len(paths)} runs", flush=True)

    for cp in paths:
        run = os.path.basename(cp)[:-4]
        out_path = os.path.join(a.out_dir, f"{run}.{a.stage}.json")
        if os.path.exists(out_path) and not a.overwrite:
            print(f"[judge] {run}: done, skipping")
            continue
        rows = list(csv.DictReader(open(cp)))
        if not rows:
            continue
        spans = [span_of(r) for r in rows]

        if a.stage == "sentiment":
            probs, embs = [], []
            for i0 in range(0, len(spans), a.batch):
                sub = spans[i0:i0 + a.batch]
                enc = jt(sub, return_tensors="pt", padding=True, truncation=True,
                         max_length=128).to(device)
                with torch.no_grad():
                    o = jm(**enc, output_hidden_states=True)
                probs.extend(torch.softmax(o.logits.float(), -1)[:, a.target_label].tolist())
                h = o.hidden_states[-1]
                m = enc["attention_mask"].unsqueeze(-1).float()
                embs.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy())
            embs = np.concatenate(embs)
            probs = np.array(probs)

            by_prompt = {}
            for r, s, e, q in zip(rows, spans, embs, probs):
                by_prompt.setdefault(r["prompt_idx"], []).append((s, e, q))
            d1 = float(np.mean([distinct_n([x[0] for x in v], 1) for v in by_prompt.values()]))
            d3 = float(np.mean([distinct_n([x[0] for x in v], 3) for v in by_prompt.values()]))
            sb = float(np.nanmean([self_bleu([x[0] for x in v]) for v in by_prompt.values()]))
            ss = float(np.nanmean([semantic_spread(np.array([x[1] for x in v]))
                                   for v in by_prompt.values()]))
            res = dict(run_name=run, stage="sentiment", judge=a.sentiment_judge,
                       n=len(rows), target_label=a.target_label,
                       adherence_pct=float(100.0 * (probs > 0.5).mean()),
                       mean_target_prob=float(probs.mean()),
                       distinct_1=d1, distinct_3=d3, self_bleu_4=sb,
                       semantic_spread=ss, csv=cp)
        else:
            nlls = []
            for s in spans:
                enc = jt(s, return_tensors="pt", truncation=True, max_length=128).to(device)
                if enc.input_ids.shape[1] < 2:
                    continue
                with torch.no_grad():
                    lg = jm(**enc).logits[0, :-1].float()
                lp = torch.log_softmax(lg, -1).gather(
                    1, enc.input_ids[0, 1:].unsqueeze(1)).squeeze(1)
                nlls.append(float(-lp.mean()))
            res = dict(run_name=run, stage="fluency", judge=a.judge_path, n=len(nlls),
                       judge_nll_per_token=float(np.mean(nlls)),
                       judge_perplexity=float(np.exp(np.mean(nlls))),
                       judge_nll_sd=float(np.std(nlls)), csv=cp)

        atomic_json(out_path, res)
        head = (f"adherence={res['adherence_pct']:.1f}% spread={res['semantic_spread']:.3f}"
                if a.stage == "sentiment" else f"ppl={res['judge_perplexity']:.1f}")
        print(f"[judge] {run}: {head}", flush=True)


if __name__ == "__main__":
    main()
