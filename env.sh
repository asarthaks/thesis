#!/usr/bin/env bash
#
# env.sh - central path anchors for the restructured repository.
#
# Workflow:  cd <repo root> && source env.sh   then run the commands in README.md.
# Sourcing this from the repo root exports every path the scripts and README rely on,
# so a future move of the tree is a single edit here.
#
# Environment reproduction:
#   exact:  gfn-lm-tuning/gfn/bin/python -m pip install -r requirements.txt.lock
#   loose:  pip install -r requirements.txt
# The active virtualenv is gfn-lm-tuning/gfn (never moved; venvs hardcode absolute paths).

# Repo root = the directory that holds this file.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export REPO_ROOT

# Grouped data / artifact roots (see README "Repository layout").
export RESULTS_DIR="$REPO_ROOT/results"
export STATUS_DIR="$REPO_ROOT/status"
export LOGS_DIR="$REPO_ROOT/logs"

# External clones / caches (moved under external/; no venv inside them).
export SEDD_REPO="$REPO_ROOT/external/Score-Entropy-Discrete-Diffusion"
export HF_HOME="$REPO_ROOT/external/hf/cache"

# Frozen base model (gpt2sft) lives inside gfn-lm-tuning/ (unchanged, left in place).
export GPT2SFT="$REPO_ROOT/gfn-lm-tuning/infill_subj_arithmetic/gpt2_large_sft_output"

# Sentiment judge head: a server-specific absolute path outside the repo tree
# (on /mount/arbeitsdaten). Edit if you run on a different machine.
export SENTIMENT_HEAD="/mount/arbeitsdaten/studenten1/singhsk/models/sentiment_constrained_ft_gpt2_large/sentiment_head.pt"

# The entry runners live in scripts/ and add the repo root to sys.path themselves,
# but exporting PYTHONPATH makes any invocation style resolve the promoted packages
# (core/, Methods/, diagnostics/, revision/) and the top-level run_experiment module.
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/scripts:$REPO_ROOT/diagnostics${PYTHONPATH:+:$PYTHONPATH}"

# SEDD datasets load with custom code (ROCStories / trajectory jobs).
export HF_DATASETS_TRUST_REMOTE_CODE=1
