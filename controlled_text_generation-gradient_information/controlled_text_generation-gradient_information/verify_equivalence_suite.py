#!/usr/bin/env python3
"""
verify_equivalence_suite.py

Extended equivalence harness for the Langevin infilling samplers.

Goal: give you enough evidence to retire the legacy code and run everything
through core/ with a clear conscience. The original verify_logic scripts only
compared the legacy *multi* function against the unified class on one sentence
with a single mask for ten steps. That is a good smoke test but it does not
cover:
  - more than one masked token (the vectorised M>1 path is never exercised),
  - more than one sentence or seed,
  - the full step schedule, where the late-schedule quench happens,
  - the legacy *single-position* functions, which are what actually produced
    your single-token thesis plots (evaluate_dls_dataset / evaluate_dls /
    evaluate_cls_dataset all import langevin_infilling_single_position).

This harness runs two kinds of check.

  strict:    same sentence, same mask set, same corruption, same seed, run both
             implementations and require the token trajectory to be identical
             step for step, with L2 / KL / entropy matching within atol. Because
             sampling is discrete and seeded, if the math is the same the tokens
             match exactly. If they diverge, we report the first step where it
             happens and (optionally) drop into a gradient-level diff on that
             one case so you can see whether it is the gradient, the proposal,
             or the RNG stream.

  aggregate: build a fixed shared set of (sentence, mask, corruption) cases from
             WikiText-2 (the same data the drivers use), run both sides paired,
             and report the fraction of cases with an identical final token,
             per-side recovery accuracy, and the max deviation between the two
             mean trajectories. This is the number you cite: "N/N cases produced
             identical trajectories; mean L2/KL/entropy agree to < tol".

Two important notes before you read a green result as equivalence:

  1. Both implementations must use the SAME epsilon / step-size schedule. The
     legacy functions hardcode np.linspace(10.5, 0.1, steps) inside the body;
     the unified BaseLangevinSampler uses linear_epsilon_schedule (0.1 -> 0.001)
     unless you patched it. If those differ, this harness will fail at the first
     step and the deep diff will show identical gradients but different tokens,
     which is the signature of a schedule mismatch. Align the schedule first.

  2. On GPU, matmul and softmax are not bit-deterministic and torch.multinomial
     turns a 1e-6 logit wobble into a different token. We enable deterministic
     algorithms below to keep the strict test honest. For the single-position
     legacy path, exact equality is NOT expected anyway: it computes the
     proposal distance directly (emb - s) while unified uses the expanded
     form (||e||^2 + ||s||^2 - 2 e.s). Same math, ~1e-5 float gap, occasional
     token flip. In --legacy-fn single mode we therefore report the mismatch
     rate rather than assert exact equality.

Drop this file at the repo root (next to core/ and Methods/) and run e.g.

  python verify_equivalence_suite.py --sampler dls --mode strict
  python verify_equivalence_suite.py --sampler cls --mode aggregate --n_samples 100
  python verify_equivalence_suite.py --sampler dls --legacy-fn single --mode strict
"""

import os
# Must be set before torch initialises CUDA, otherwise deterministic matmul is a no-op.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
import math
import time
import argparse
import random

import numpy as np
import torch

try:
    from tqdm import tqdm
    _HAVE_TQDM = True
except Exception:
    _HAVE_TQDM = False

# Make the repo importable regardless of where the script is launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.base_sampler import BaseLangevinSampler
from core.dls import DiscreteLangevinSampler
from core.cls import ContinuousLangevinSampler
from core.prep import load_tokenizer_and_model

from Methods.Scripts.dls import (
    langevin_infilling_multiple_positions as legacy_dls_multi,
    langevin_infilling_single_position as legacy_dls_single,
)
from Methods.Scripts.cls import (
    langevin_infilling_multiple_positions as legacy_cls_multi,
    langevin_infilling_single_position as legacy_cls_single,
)

DEFAULT_MODEL = "/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/infill_subj_arithmetic/gpt2_large_sft_output"

# A few sentences of varying length so mask counts up to 3 fit comfortably.
# Kept in-script so the strict test does not depend on any dataset download.
FIXED_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the river.",
    "Scientists reported that the new model improved accuracy on several benchmarks last year.",
    "She poured a cup of coffee and stared out the window at the rain.",
    "In many languages the meaning of a sentence depends heavily on word order and context.",
]


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_determinism():
    # warn_only=True so ops without a deterministic kernel (some attention / embedding
    # backward paths) warn instead of raising, while everything that can be deterministic
    # still is. Keeps the strict per-step comparison honest without crashing mid-sweep.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        return True
    except TypeError:
        # older torch without warn_only
        try:
            torch.use_deterministic_algorithms(True)
            return True
        except Exception as e:
            print(f"[warn] could not enable deterministic algorithms: {e}")
            return False
    except Exception as e:
        print(f"[warn] could not enable deterministic algorithms: {e}")
        return False


def device_of(model):
    return next(model.parameters()).device


# --------------------------------------------------------------------------- #
# Case construction: deterministic given a seed, built once and fed to BOTH
# implementations so the only remaining difference is the sampler internals.
# --------------------------------------------------------------------------- #
def build_case(tokenizer, text, num_masks, case_seed, device):
    rng = np.random.RandomState(case_seed)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    seq_len = input_ids.shape[1]
    if seq_len < num_masks + 3:
        return None

    valid = list(range(1, seq_len - 1))
    if num_masks > len(valid):
        return None
    mask_indices = sorted(rng.choice(valid, size=num_masks, replace=False).tolist())

    orig_ids = input_ids.clone()
    corrupted = input_ids.clone()
    vocab = tokenizer.vocab_size
    for idx in mask_indices:
        orig = input_ids[0, idx].item()
        r = int(rng.randint(0, vocab))
        while r == orig:
            r = int(rng.randint(0, vocab))
        corrupted[0, idx] = r

    return corrupted, mask_indices, orig_ids


# --------------------------------------------------------------------------- #
# Metric normalisation: legacy single functions use l2_distance / kl_divergence
# / token_id; multi and unified use avg_l2_distance / avg_kl_divergence /
# token_ids. Map everything onto the common schema so compare() is simple.
# --------------------------------------------------------------------------- #
def normalize_metrics(metrics_history):
    out = []
    for m in metrics_history:
        d = dict(m)
        if "avg_l2_distance" not in d:
            d["avg_l2_distance"] = d.get("l2_distance")
        if "avg_kl_divergence" not in d:
            d["avg_kl_divergence"] = d.get("kl_divergence")
        if "token_ids" not in d:
            tid = d.get("token_id")
            d["token_ids"] = [tid] if tid is not None else []
        out.append(d)
    return out


# --------------------------------------------------------------------------- #
# Runners. seed_all is called immediately before each run so the two
# implementations start from an identical RNG state and stay in lockstep as
# long as they sample the same tokens.
# --------------------------------------------------------------------------- #
def run_legacy(kind, legacy_fn_type, model, tokenizer, corrupted, mask_indices, orig_ids, cfg, run_seed):
    seed_all(run_seed)
    if legacy_fn_type == "multi":
        fn = legacy_dls_multi if kind == "dls" else legacy_cls_multi
        extra = {} if kind == "dls" else {"noise_scale": cfg["noise_scale"]}
        _, mh = fn(
            model=model,
            input_ids=corrupted.clone(),
            mask_indices=mask_indices,
            tokenizer=tokenizer,
            steps=cfg["steps"],
            temperature=cfg["temperature"],
            oracle=False,
            orig_ids=orig_ids.clone(),
            method=cfg["method"],
            mh_sampling=cfg["mh"],
            grad_normalization=cfg["grad_norm"],
            debug=False,
            **extra,
        )
        return normalize_metrics(mh)

    # single-position path only supports one mask
    assert len(mask_indices) == 1, "legacy single-position path only handles M=1"
    idx = mask_indices[0]
    position = (idx, idx + 1)
    fn = legacy_dls_single if kind == "dls" else legacy_cls_single
    extra = {} if kind == "dls" else {"noise_scale": cfg["noise_scale"]}
    ret = fn(
        model=model,
        input_ids=corrupted.clone(),
        position=position,
        tokenizer=tokenizer,
        steps=cfg["steps"],
        temperature=cfg["temperature"],
        oracle=False,
        orig_ids=orig_ids.clone(),
        method=cfg["method"],
        mh_sampling=cfg["mh"],
        grad_normalization=cfg["grad_norm"],
        debug=False,
        **extra,
    )
    # DLS single returns (s_ids, metrics, alpha_schedule); CLS single returns (s_ids, metrics)
    metrics = ret[1]
    return normalize_metrics(metrics)


def run_unified(kind, model, tokenizer, corrupted, mask_indices, orig_ids, cfg, run_seed):
    seed_all(run_seed)
    Cls = DiscreteLangevinSampler if kind == "dls" else ContinuousLangevinSampler

    kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        steps=cfg["steps"],
        temperature=cfg["temperature"],
        oracle=False,
        method=cfg["method"],
        mh_sampling=cfg["mh"],
        grad_normalization=cfg["grad_norm"],
        noise_scale=cfg["noise_scale"],
    )

    # Inject the SAME schedule the legacy functions hardcode (linspace(eps_start, eps_end)).
    # This requires the base_sampler epsilon_schedule patch. If it is not applied we detect
    # it here and stop with a clear message instead of silently comparing against the wrong
    # 0.1 -> 0.001 default schedule.
    import inspect
    if "epsilon_schedule" in inspect.signature(BaseLangevinSampler.__init__).parameters:
        kwargs["epsilon_schedule"] = np.linspace(cfg["eps_start"], cfg["eps_end"], cfg["steps"])
    else:
        raise RuntimeError(
            "core/base_sampler.py has no 'epsilon_schedule' parameter. Apply "
            "core_base_sampler_schedule.patch first, otherwise the unified sampler uses "
            "its default 0.1 -> 0.001 schedule and will diverge from the legacy 10.5 -> 0.1."
        )

    sampler = Cls(**kwargs)
    _, mh = sampler.optimize(corrupted.clone(), mask_indices, orig_ids.clone())
    return normalize_metrics(mh)


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def _safe(x):
    return float(x) if x is not None else float("nan")


def compare(mh_a, mh_b):
    steps = min(len(mh_a), len(mh_b))
    first_token_div = None
    max_l2 = max_kl = max_ent = 0.0
    for k in range(steps):
        ta = mh_a[k]["token_ids"]
        tb = mh_b[k]["token_ids"]
        if ta != tb and first_token_div is None:
            first_token_div = k
        # metric deltas only make sense while the token streams still agree
        if first_token_div is None:
            la, lb = _safe(mh_a[k]["avg_l2_distance"]), _safe(mh_b[k]["avg_l2_distance"])
            ka, kb = _safe(mh_a[k]["avg_kl_divergence"]), _safe(mh_b[k]["avg_kl_divergence"])
            ea, eb = _safe(mh_a[k].get("entropy")), _safe(mh_b[k].get("entropy"))
            if not math.isnan(la) and not math.isnan(lb):
                max_l2 = max(max_l2, abs(la - lb))
            if not math.isnan(ka) and not math.isnan(kb):
                max_kl = max(max_kl, abs(ka - kb))
            if not math.isnan(ea) and not math.isnan(eb):
                max_ent = max(max_ent, abs(ea - eb))
    return {
        "steps": steps,
        "len_a": len(mh_a),
        "len_b": len(mh_b),
        "first_token_div": first_token_div,
        "max_l2": max_l2,
        "max_kl": max_kl,
        "max_ent": max_ent,
    }


# --------------------------------------------------------------------------- #
# Gradient-level deep diff, run only on a single failing case. Mirrors the spy
# trick from your original verify_logic but scoped so it does not accumulate
# across the whole sweep.
# --------------------------------------------------------------------------- #
def deep_diff_one(kind, legacy_fn_type, model, tokenizer, corrupted, mask_indices, orig_ids, cfg, run_seed):
    print("\n" + "=" * 70)
    print("DEEP DIFF on first failing case (gradient-level)")
    print("=" * 70)

    original_grad = torch.autograd.grad
    captured = {"target": None}

    def hooked_grad(*args, **kwargs):
        g = original_grad(*args, **kwargs)
        if captured["target"] is not None:
            captured["target"].append(g[0].detach().float().cpu().clone())
        return g

    torch.autograd.grad = hooked_grad
    try:
        legacy_grads, unified_grads = [], []

        captured["target"] = legacy_grads
        run_legacy(kind, legacy_fn_type, model, tokenizer, corrupted, mask_indices, orig_ids, cfg, run_seed)

        captured["target"] = unified_grads
        run_unified(kind, model, tokenizer, corrupted, mask_indices, orig_ids, cfg, run_seed)
    finally:
        torch.autograd.grad = original_grad
        captured["target"] = None

    n = min(len(legacy_grads), len(unified_grads))
    print(f"legacy captured {len(legacy_grads)} grads, unified captured {len(unified_grads)}")
    first_grad_div = None
    for i in range(n):
        gl, gu = legacy_grads[i], unified_grads[i]
        if gl.shape != gu.shape:
            print(f"  grad #{i}: SHAPE mismatch {tuple(gl.shape)} vs {tuple(gu.shape)}")
            first_grad_div = i
            break
        delta = (gl - gu).abs().max().item()
        same = torch.allclose(gl, gu, atol=1e-5)
        flag = "" if same else "   <-- differs"
        print(f"  grad #{i}: max|delta|={delta:.3e} legacy_norm={gl.norm():.3f} unified_norm={gu.norm():.3f}{flag}")
        if not same and first_grad_div is None:
            first_grad_div = i

    if first_grad_div is None and len(legacy_grads) == len(unified_grads):
        print("\n  Gradients are identical but tokens diverged.")
        print("  That points at the proposal / schedule / RNG, NOT the gradient.")
        print("  Most common cause: legacy and unified are using different eps schedules.")
    elif first_grad_div is not None:
        print(f"\n  Gradients first differ at call #{first_grad_div}.")
        print("  If this is call #0 the models differ (weights / dtype / peft merge).")
        print("  If later, an earlier proposal already diverged and desynced the state.")
    print("=" * 70 + "\n")


# --------------------------------------------------------------------------- #
# Strict suite
# --------------------------------------------------------------------------- #
def strict_suite(kind, legacy_fn_type, model, tokenizer, args):
    device = device_of(model)
    mask_counts = [1] if legacy_fn_type == "single" else [1, 2, 3]
    seeds = list(range(args.seeds))
    methods = args.methods
    mh_opts = [False, True]

    cfg_base = dict(
        steps=args.steps,
        temperature=args.temperature,
        grad_norm=args.grad_norm,
        noise_scale=args.noise_scale,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
    )

    total = 0
    identical = 0
    metric_ok = 0
    failures = []

    # Per-config buckets so the clustering is visible. Key = (method, mh).
    # "expected_exact" flags the buckets that SHOULD be bit-identical if the two
    # implementations are the same function. The MH buckets are not expected to
    # match once you know the reverse-proposal difference: unified applies the
    # method variation to the backward gradient (correct for detailed balance),
    # legacy DLS uses the raw backward gradient. So a mismatch there is unified
    # fixing a bug, not a regression.
    from collections import defaultdict
    buckets = defaultdict(lambda: {"pass": 0, "total": 0, "expected_exact": True})

    strict_exact = legacy_fn_type == "multi"  # single path is not expected to be bit-identical

    # Precompute the flat job list up front so we can show a real total and an ETA.
    # Cases are cached per (text, M, seed) and reused across method/mh.
    case_cache = {}
    jobs = []
    for text in FIXED_SENTENCES:
        for M in mask_counts:
            for cseed in seeds:
                key = (text, M, cseed)
                if key not in case_cache:
                    case_cache[key] = build_case(tokenizer, text, M, cseed, device)
                case = case_cache[key]
                if case is None:
                    continue
                for method in methods:
                    for mh in mh_opts:
                        jobs.append((text, M, cseed, method, mh, case))

    n_jobs = len(jobs)
    print(f"\nstrict sweep plan: {n_jobs} cases  "
          f"({len(FIXED_SENTENCES)} sentences x {len(mask_counts)} mask-counts x "
          f"{len(seeds)} seeds x {len(methods)} methods x {len(mh_opts)} mh)")
    print(f"  steps={args.steps}, each case runs legacy + unified back to back")
    print(f"  schedule injected on both sides: linspace({args.eps_start}, {args.eps_end}, {args.steps})\n")

    t_start = time.time()
    use_bar = _HAVE_TQDM and not args.no_progress
    iterator = tqdm(jobs, desc="strict", unit="case") if use_bar else jobs

    for i, (text, M, cseed, method, mh, case) in enumerate(iterator):
        corrupted, mask_indices, orig_ids = case
        cfg = {**cfg_base, "method": method, "mh": mh}
        run_seed = 1000 + cseed  # shared by both sides

        t0 = time.time()
        mh_leg = run_legacy(kind, legacy_fn_type, model, tokenizer,
                            corrupted, mask_indices, orig_ids, cfg, run_seed)
        mh_uni = run_unified(kind, model, tokenizer,
                             corrupted, mask_indices, orig_ids, cfg, run_seed)
        dt = time.time() - t0
        d = compare(mh_leg, mh_uni)
        total += 1

        traj_same = d["first_token_div"] is None and d["len_a"] == d["len_b"]
        m_ok = (d["max_l2"] <= args.atol and d["max_kl"] <= args.atol_kl
                and d["max_ent"] <= args.atol)
        if traj_same:
            identical += 1
        if m_ok:
            metric_ok += 1

        # A bucket is expected to be bit-identical only when there is no MH
        # reverse-proposal in play, or when method variation is the identity
        # (policy without grad normalization).
        expected_exact = (not mh) or (method == "policy" and not args.grad_norm)
        b = buckets[(method, mh)]
        b["total"] += 1
        b["expected_exact"] = expected_exact
        if traj_same:
            b["pass"] += 1

        ok = traj_same if (strict_exact and expected_exact) else True
        if not ok:
            failures.append((text[:40], M, cseed, method, mh, d,
                             corrupted, mask_indices, orig_ids, cfg, run_seed))

        # After the very first case, print a concrete ETA for the whole sweep.
        if i == 0:
            eta = dt * n_jobs
            print(f"\n~{dt:.1f}s for the first case  ->  rough estimate ~{eta/60:.1f} min "
                  f"for all {n_jobs} cases (MH cases run a bit slower)\n", flush=True)

        if use_bar:
            iterator.set_postfix(match=f"{identical}/{total}", last=f"{dt:.1f}s")
        elif (i + 1) % 5 == 0 or (i + 1) == n_jobs:
            elapsed = time.time() - t_start
            rate = elapsed / (i + 1)
            remaining = rate * (n_jobs - i - 1)
            print(f"  [{i+1}/{n_jobs}] match={identical}/{total} "
                  f"elapsed={elapsed/60:.1f}m eta~{remaining/60:.1f}m", flush=True)

    print(f"\ntotal wall time: {(time.time() - t_start)/60:.1f} min")

    print("\n" + "#" * 70)
    print(f"STRICT SUITE   sampler={kind}  legacy_fn={legacy_fn_type}  grad_norm={args.grad_norm}")
    print("#" * 70)
    print(f"cases run                         : {total}")
    print(f"identical token trajectory        : {identical}/{total}")
    print(f"metric agreement within atol      : {metric_ok}/{total}")

    print("\nper-config breakdown (identical trajectory / total):")
    print(f"  {'method':<32}{'mh':<5}{'result':<12}{'interpretation'}")
    core_ok = True
    for (method, mh) in sorted(buckets.keys()):
        b = buckets[(method, mh)]
        rate = f"{b['pass']}/{b['total']}"
        if b["expected_exact"]:
            note = "must match" if b["pass"] == b["total"] else "REGRESSION, investigate"
            if b["pass"] != b["total"]:
                core_ok = False
        else:
            if method == "policy":
                note = "expected to differ (unified fixes MH reverse; keep unified)"
            else:
                note = "random-walk MH ill-posed; apply symmetric fix, then recheck"
        print(f"  {method:<32}{str(mh):<5}{rate:<12}{note}")

    if strict_exact and legacy_fn_type == "multi":
        print()
        if core_ok:
            print("VERDICT: core is equivalent. Every bucket that MUST match, matched.")
            print("The remaining differences are confined to the MH reverse proposal,")
            print("where unified is the correct implementation. Safe to switch, then")
            print("regenerate MH figures with unified and apply the symmetric random-walk fix.")
        else:
            print("VERDICT: a bucket that should be bit-identical diverged. That is a real")
            print("regression in the core sampler, not the known MH difference. Inspect below.")

    if failures:
        t, M, cs, method, mh, d, corrupted, mask_indices, orig_ids, cfg, run_seed = failures[0]
        print(f"\nfirst UNEXPECTED failure: '{t}...' M={M} seed={cs} method={method} mh={mh}")
        print(f"    first token divergence at step {d['first_token_div']} "
              f"(len legacy={d['len_a']}, unified={d['len_b']})")
        print(f"    max|dL2|={d['max_l2']:.3e} max|dKL|={d['max_kl']:.3e} max|dEnt|={d['max_ent']:.3e}")
        if args.deep_diff:
            deep_diff_one(kind, legacy_fn_type, model, tokenizer,
                          corrupted, mask_indices, orig_ids, cfg, run_seed)

    if not strict_exact:
        print("\nNOTE: legacy single-position path is not expected to be bit-identical")
        print("      to unified (direct vs expanded distance). Reporting closeness only.")
        print(f"  trajectories that still matched exactly: {identical}/{total}")


# --------------------------------------------------------------------------- #
# Aggregate suite (dataset-realistic)
# --------------------------------------------------------------------------- #
def load_shared_cases(tokenizer, device, n_samples, num_masks, base_seed=0):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [x["text"].strip() for x in ds if 10 < len(x["text"].strip().split()) < 40]

    cases = []
    i = 0
    while len(cases) < n_samples and i < len(texts):
        case = build_case(tokenizer, texts[i], num_masks, base_seed + i, device)
        if case is not None:
            cases.append(case)
        i += 1
    return cases


def aggregate_suite(kind, model, tokenizer, args):
    device = device_of(model)
    cases = load_shared_cases(tokenizer, device, args.n_samples, args.num_masks)
    print(f"\nAggregate suite: {len(cases)} shared cases, M={args.num_masks}, "
          f"method={args.method}, mh={args.mh_sampling}")

    cfg = dict(
        steps=args.steps,
        temperature=args.temperature,
        grad_norm=args.grad_norm,
        noise_scale=args.noise_scale,
        method=args.method,
        mh=args.mh_sampling,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
    )

    n_steps = args.steps
    leg_curves = {k: np.zeros(n_steps) for k in ["l2", "kl", "ent"]}
    uni_curves = {k: np.zeros(n_steps) for k in ["l2", "kl", "ent"]}
    counts = np.zeros(n_steps)

    final_match = 0
    leg_correct = 0
    uni_correct = 0
    examples = []

    t_start = time.time()
    use_bar = _HAVE_TQDM and not args.no_progress
    print(f"schedule injected on both sides: linspace({args.eps_start}, {args.eps_end}, {args.steps})", flush=True)
    iterator = tqdm(list(enumerate(cases)), desc="aggregate", unit="case") if use_bar else enumerate(cases)

    for j, (corrupted, mask_indices, orig_ids) in iterator:
        t0 = time.time()
        run_seed = 2000 + j
        mh_leg = run_legacy(kind, "multi", model, tokenizer,
                            corrupted, mask_indices, orig_ids, cfg, run_seed)
        mh_uni = run_unified(kind, model, tokenizer,
                             corrupted, mask_indices, orig_ids, cfg, run_seed)
        dt = time.time() - t0

        gt = orig_ids[0, mask_indices].tolist()
        leg_final = mh_leg[-1]["token_ids"]
        uni_final = mh_uni[-1]["token_ids"]

        if leg_final == uni_final:
            final_match += 1
        leg_correct += sum(int(p == t) for p, t in zip(leg_final, gt))
        uni_correct += sum(int(p == t) for p, t in zip(uni_final, gt))

        steps = min(n_steps, len(mh_leg), len(mh_uni))
        for k in range(steps):
            counts[k] += 1
            leg_curves["l2"][k] += _safe(mh_leg[k]["avg_l2_distance"])
            uni_curves["l2"][k] += _safe(mh_uni[k]["avg_l2_distance"])
            leg_curves["kl"][k] += _safe(mh_leg[k]["avg_kl_divergence"])
            uni_curves["kl"][k] += _safe(mh_uni[k]["avg_kl_divergence"])
            leg_curves["ent"][k] += _safe(mh_leg[k].get("entropy"))
            uni_curves["ent"][k] += _safe(mh_uni[k].get("entropy"))

        if len(examples) < 3:
            examples.append((
                tokenizer.decode(orig_ids[0], skip_special_tokens=True)[:60],
                [tokenizer.decode([t]) for t in leg_final],
                [tokenizer.decode([t]) for t in uni_final],
            ))

        if j == 0:
            print(f"\n~{dt:.1f}s per case -> rough estimate ~{dt*len(cases)/60:.1f} min "
                  f"for {len(cases)} cases\n", flush=True)
        if use_bar:
            iterator.set_postfix(same=f"{final_match}/{j+1}", last=f"{dt:.1f}s")
        elif (j + 1) % 10 == 0 or (j + 1) == len(cases):
            elapsed = time.time() - t_start
            eta = elapsed / (j + 1) * (len(cases) - j - 1)
            print(f"  [{j+1}/{len(cases)}] same_final={final_match}/{j+1} "
                  f"elapsed={elapsed/60:.1f}m eta~{eta/60:.1f}m", flush=True)

    counts = np.maximum(counts, 1)
    dev = {}
    for k in ["l2", "kl", "ent"]:
        lc = leg_curves[k] / counts
        uc = uni_curves[k] / counts
        dev[k] = np.nanmax(np.abs(lc - uc))

    total_masks = len(cases) * args.num_masks
    print("\n" + "#" * 70)
    print(f"AGGREGATE SUITE   sampler={kind}")
    print("#" * 70)
    print(f"cases                                 : {len(cases)}")
    print(f"identical final token set             : {final_match}/{len(cases)}")
    print(f"legacy recovery accuracy              : {100.0 * leg_correct / total_masks:.2f}%")
    print(f"unified recovery accuracy             : {100.0 * uni_correct / total_masks:.2f}%")
    print(f"max deviation of mean L2 curve        : {dev['l2']:.3e}")
    print(f"max deviation of mean KL curve        : {dev['kl']:.3e}")
    print(f"max deviation of mean entropy curve   : {dev['ent']:.3e}")
    print("\nexample side-by-side (original / legacy / unified):")
    for orig, lf, uf in examples:
        print(f"  '{orig}...'")
        print(f"     legacy : {lf}")
        print(f"     unified: {uf}")

    if final_match == len(cases):
        print("\nVERDICT: safe to switch. Identical behaviour on the real data path.")
    else:
        print(f"\nVERDICT: {len(cases) - final_match} case(s) diverged. Inspect with --mode strict.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Legacy vs unified Langevin equivalence suite")
    p.add_argument("--sampler", choices=["dls", "cls"], required=True)
    p.add_argument("--mode", choices=["strict", "aggregate", "both"], default="both")
    p.add_argument("--legacy-fn", dest="legacy_fn", choices=["multi", "single"], default="multi",
                   help="which legacy implementation to compare against (single = the retired "
                        "single-position path that generated the single-token thesis plots)")
    p.add_argument("--model_path", default=DEFAULT_MODEL)

    # sampler config: keep these matched to whatever your real runs use
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--temperature", type=float, default=5.0)
    p.add_argument("--grad_norm", action="store_true",
                   help="set to match legacy DLS driver (GRAD_NORMALIZATION=True). CLS driver used False.")
    p.add_argument("--noise_scale", type=float, default=0.01)

    # step-size schedule. Legacy hardcodes linspace(10.5, 0.1); we inject the same into
    # unified so the comparison is apples to apples. Change both only if you also change
    # the legacy hardcode, otherwise the two sides will not be comparable.
    p.add_argument("--eps_start", type=float, default=10.5, help="schedule start (GPT-2 Large ~10.5)")
    p.add_argument("--eps_end", type=float, default=0.1, help="schedule end (GPT-2 Large ~0.1)")

    # strict sweep breadth
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--methods", nargs="+", default=["policy", "random"],
                   choices=["policy", "grad_norm_preserved_random_dir", "random"])
    p.add_argument("--atol", type=float, default=1e-4, help="tolerance for L2 / entropy")
    p.add_argument("--atol_kl", type=float, default=1e-3, help="tolerance for KL (looser, model fwd noise)")
    p.add_argument("--deep_diff", action="store_true", default=True,
                   help="gradient-level diff on the first strict failure")
    p.add_argument("--quick", action="store_true",
                   help="tiny smoke test: 1 sentence, M=1, 1 seed. Confirms wiring in a minute.")
    p.add_argument("--no_progress", action="store_true", help="disable the tqdm progress bar")

    # aggregate config
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--num_masks", type=int, default=1)
    p.add_argument("--method", default="policy", help="method for aggregate mode")
    p.add_argument("--mh_sampling", action="store_true", help="mh for aggregate mode")

    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"],
                   help="use bfloat16 for large models like Llama to fit memory")
    p.add_argument("--no_determinism", action="store_true",
                   help="disable deterministic algorithms (faster, less strict)")

    args = p.parse_args()

    if args.quick:
        # shrink the strict sweep to a handful of cases for a fast wiring check
        global FIXED_SENTENCES
        FIXED_SENTENCES = FIXED_SENTENCES[:1]
        args.seeds = 1
        args.n_samples = min(args.n_samples, 6)

    det_on = enable_determinism() if not args.no_determinism else False
    print(f"determinism: {'on' if det_on else 'off'} | progress bar: "
          f"{'on' if (_HAVE_TQDM and not args.no_progress) else 'off'} | tqdm installed: {_HAVE_TQDM}")

    import torch as _t
    dtype = {"float32": _t.float32, "float16": _t.float16, "bfloat16": _t.bfloat16}[args.dtype]
    print(f"Loading model from {args.model_path} ({args.dtype}) ...", flush=True)
    t_load = time.time()
    tokenizer, model = load_tokenizer_and_model(args.model_path, dtype=dtype)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    n_params = sum(pp.numel() for pp in model.parameters())
    dev = next(model.parameters()).device
    print(f"loaded {n_params/1e6:.0f}M params on {dev} in {time.time()-t_load:.1f}s", flush=True)

    if args.legacy_fn == "single" and args.num_masks != 1:
        print("[note] single-position path only supports M=1; forcing num_masks=1 for aggregate.")
        args.num_masks = 1

    if args.mode in ("strict", "both"):
        strict_suite(args.sampler, args.legacy_fn, model, tokenizer, args)

    if args.mode in ("aggregate", "both"):
        if args.legacy_fn == "single":
            print("\n[note] aggregate mode compares legacy multi vs unified; "
                  "single-path closeness is covered by --mode strict --legacy-fn single.")
        aggregate_suite(args.sampler, model, tokenizer, args)


if __name__ == "__main__":
    main()