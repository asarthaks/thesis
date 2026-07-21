#!/usr/bin/env python
"""
sedd_lib.py

Shared SEDD helpers for the Phase 4 capability pilots (Part 0 gates, P1 native
recovery, H hybrid sampler, G guided generation). One place for the model load and
the two SEDD read-outs, so every pilot uses the SAME verified interface.

Facts established in Phase 2/3 (REVISION_LOG.md) and re-checked here:
  - SEDD is ABSORBING, dim 50258 = 50257 real GPT-2 tokens + one MASK id (50257).
  - The raw model output model(ids, sigma) is ALREADY in log space (a score).
  - The score at a position is context-dependent ONLY when that position is MASK.
  - Conditional infilling (run_sample_cond.py): get_pc_sampler with a proj_fun that
    clamps observed positions to their true tokens every step; masked positions start
    at MASK (graph.sample_limit) and are denoised; the denoiser truncates the MASK
    column so a committed fill is always a real vocabulary token.

Nothing here touches core/ or the SEDD clone; the clone is only imported.
"""

import os
import sys

import torch
import torch.nn.functional as F

MASK_TOKEN = 50257
REAL_VOCAB = 50257

SEDD_REPO = os.environ.get(
    "SEDD_REPO",
    "/mount/studenten-temp1/users/singhsk/thesis/thesis/Score-Entropy-Discrete-Diffusion")
# route the HF cache to where sedd-small already lives unless the caller overrode it
os.environ.setdefault(
    "HF_HOME",
    "/mount/studenten-temp1/users/singhsk/thesis/thesis/hf/cache")


class SEDDBundle:
    def __init__(self, model, graph, noise, tokenizer, device, scale):
        self.model = model
        self.graph = graph
        self.noise = noise
        self.tokenizer = tokenizer
        self.device = device
        self.scale = scale


def load_sedd(scale, device="cuda"):
    """scale in {'small','medium'} -> louaaron/sedd-<scale> via the patched clone."""
    if SEDD_REPO not in sys.path:
        sys.path.insert(0, SEDD_REPO)
    from load_model import load_model                # noqa: E402
    from transformers import GPT2TokenizerFast       # noqa: E402
    model_id = f"louaaron/sedd-{scale}"
    model, graph, noise = load_model(model_id, device)
    model.eval()
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    return SEDDBundle(model, graph, noise, tok, device, scale)


@torch.no_grad()
def score_logspace(bundle, ids_BL, sigma):
    """Raw SEDD score, shape (B, L, dim), ALREADY log space. ids_BL: B x L Long."""
    sig = torch.tensor([sigma] * ids_BL.shape[0], device=bundle.device)
    return bundle.model(ids_BL.to(bundle.device), sig).float()


@torch.no_grad()
def logpref_at(bundle, ids_1L, pos, sigma):
    """One-pass SEDD denoising log-preference over the REAL vocab at `pos`.
    Masks `pos`, keeps the rest as given (observed context), one forward,
    log_softmax over the real vocabulary (MASK column dropped). Returns (V_real,).
    This is the surrogate readout (linearization) and the hybrid proposal readout."""
    x = ids_1L.clone()
    x[0, pos] = MASK_TOKEN
    s = score_logspace(bundle, x, sigma)[0, pos, :REAL_VOCAB]
    return F.log_softmax(s, dim=-1)


@torch.no_grad()
def conditional_recovery(bundle, ids_1L, mask_positions, predictor="euler",
                         steps=128, denoise=True, eps=1e-5, seed=None,
                         tail=0):
    """Native SEDD absorbing conditional denoising.

    ids_1L: 1 x L clean sequence (the true observed context; the tokens at
            mask_positions are ignored, those positions are re-generated).
    mask_positions: list[int] positions to fill (start at MASK, denoised).
    tail: append `tail` extra MASK positions after the sequence (for the Part 0
          tail-leakage gate). Tail positions are NOT clamped and NOT reported.

    Returns the recovered 1 x L sequence (tail stripped). Observed positions are
    clamped every step, so they are bit-identical on return; fills are real vocab.
    """
    import sampling  # from the SEDD clone (already on sys.path via load_sedd)

    device = bundle.device
    L = ids_1L.shape[1]
    obs_locs = [i for i in range(L) if i not in set(mask_positions)]
    obs_ids = ids_1L[0, obs_locs].clone().to(device)
    total_len = L + tail

    def proj_fun(x):
        x[:, obs_locs] = obs_ids
        return x

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    fn = sampling.get_pc_sampler(bundle.graph, bundle.noise, (1, total_len),
                                 predictor, steps, denoise=denoise, eps=eps,
                                 device=device, proj_fun=proj_fun)
    out = proj_fun(fn(bundle.model))
    return out[:, :L]
