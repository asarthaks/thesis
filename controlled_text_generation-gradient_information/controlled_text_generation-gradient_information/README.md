# Diagnostics: how to run

Everything here answers one question: **why doesn't it work.** Nothing here samples.

---

## Step 0. Install the patched samplers

```bash
cp core/dls.py core/dls.py.bak
cp core/cls.py core/cls.py.bak
cp diagnostics/core_patched/dls.py core/dls.py
cp diagnostics/core_patched/cls.py core/cls.py
```

The patched files add three optional recorders, all `None` by default:

| attribute | records | used by |
|---|---|---|
| `sampler.mh_log` | one dict per MH decision, with the acceptance ratio **decomposed** into its target term and its proposal term | EXP 2 |
| `sampler.traj_log` | the raw state at every step, plus its distance to the nearest token embedding | EXP 3 |
| `sampler.proposal_log` | (DLS only) the relative scale of the distance term `t1` versus the gradient term `t2` in the proposal logits | EXP 1, supporting |

When all three are `None` the files are **bit-identical** to the originals. No extra
tensors, no extra `torch.randn`, so the RNG stream stays aligned and your 145 existing
runs remain reproducible. Verify with your own equivalence suite before trusting me:

```bash
python verify_equivalence_suite.py     # should still pass every must-match bucket
```

---

## Step 1. The queued experiments (EXP 1, 4, 5)

Ten jobs. Same contract as `run_experiment.py`: done when the JSON exists, atomic
write, resume by existence. `run_queue.sh`, `worker.sh` and `reset_incomplete.sh`
need no changes.

```bash
python diagnostics/gen_diag_manifest.py --out_dir results_diag --max_vram 24 \
    > manifest_diag.tsv

./run_queue.sh --manifest manifest_diag.tsv --gpus "0 1 2 3 4 5 6 7" --per_gpu 2 \
    --vram 24 --out_dir results_diag --status status_diag --env gfn
```

Edit the `MODELS` dict at the top of `gen_diag_manifest.py` first: the adapter paths
are guesses.

Only `linearization` is heavy (a few hours per model, 200 sequences x 2000 candidates).
`anisotropy` is two minutes. `likelihood_trap` is maybe twenty.

---

## Step 2. The trace experiments (EXP 2, 3)

Not queued, because they need the patched samplers and they are a single run each.

```bash
python diagnostics/collect_traces.py \
    --model_path ./gpt2_sft_output \
    --core_path . \
    --run_name traces_gpt2sft \
    --out_dir results_diag \
    --n_seqs 200 --n_traj_seqs 6 --steps 50 --n_masks 1
```

**Keep `--n_masks 1`.** With more than one masked position the MH accept/reject is a
single joint decision over all of them, so "did the proposal cross a boundary" stops
being a clean binary and the conditional acceptance rate becomes uninterpretable. Your
thesis already establishes that the qualitative behaviour is preserved for M > 1, so
M = 1 is the correct setting for a mechanism measurement.

`collect_traces.py` introspects `BaseLangevinSampler` rather than assuming a signature
I cannot see. If it raises, the error message names the exact function to fix, and
there are only two: `build_sampler()` and `call_run()`. Both are at the top of the
file under a marked ADAPT block.

---

## Step 3. Figures

```bash
python diagnostics/plot_diagnostics.py --res_dir results_diag --fig_dir figures
python diagnostics/analyze_mh.py \
    --csv results_diag/traces_gpt2sft_mh.csv --fig_dir figures
```

Outputs, in the order they appear in the mechanism chapter:

| figure | what it establishes |
|---|---|
| `fig_lin_radius` | **THE ONE.** The gradient is informative only within a radius far smaller than the smallest possible token swap. |
| `fig_lin_scatter` | The Taylor surrogate is uncorrelated with the true energy change. |
| `fig_lin_decomposition` | The surrogate is structurally blind to `log p(x_i \| x_<i)`, which is the larger of the two terms. |
| `fig_lin_topk` | Ranking candidate tokens by the gradient is no better than ranking them at random. |
| `fig_mh_accept` | MH accepts only the proposals that change nothing. |
| `fig_mh_decomposition` | The rejection comes from the **proposal** term, not the target term. Boundary-crossing moves often *improve* the sequence and are rejected anyway. |
| `fig_mh_dls_vs_cls` | The same correction, same model, two state spaces, opposite outcomes. |
| `fig_traj_pca` / `fig_traj_tsne` | Where the sampler actually goes. CLS with grad-norm never leaves its cell; CLS without it leaves the token manifold entirely. |
| `fig_traj_manifold` | How far off the manifold, quantitatively. The LM is being asked to score inputs it has never seen. |
| `fig_trap_scatter` | Lowest energy = worst text. The likelihood trap, on your model. |
| `fig_trap_length` | The unnormalised GFlowNet reward is a length penalty in disguise. The slope is the brevity incentive, in nats per token. |
| `fig_aniso_hist` | 4.5 versus 0.5. Verifies the number you cite in three places. |

---

## Step 4. Send me two files

```
figures/diagnostic_summary.csv
figures/mh_summary.csv
```

Those contain every number the mechanism chapter needs, and I will write the chapter
against them rather than against my predictions.

---

## Restoring production behaviour

```bash
cp core/dls.py.bak core/dls.py
cp core/cls.py.bak core/cls.py
```

Not strictly necessary, since the recorders are inert when unset, but keep the backups.
