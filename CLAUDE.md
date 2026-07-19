# CLAUDE.md — Flavours of Misalignment (this subdirectory)

Context for a Claude session grading or reproducing this project. This file
is scoped to this repo (`AISafety/`), a paper reproduction codebase — not to
be confused with a parent-directory `CLAUDE.md` for an unrelated course, if
one exists above this directory.

## What this is

Code accompanying the paper **"Flavours of Misalignment"**
(`pdf_Flavours_of_Misalignment.pdf`). It builds three intervention variants
from a base LLM (emergent-misalignment finetune on bad data, the same on
good data as a control, and refusal-direction ablation), evaluates them
(and their pairwise stacks) on capability/safety/OOD benchmarks, and
analyzes where the refusal concept lives in their activations (cosine
similarity to the base model's refusal vector, layer-wise CKA).

## Start here

**`README.md` is the source of truth for every command** — repo structure,
setup, and how to run each stage of the pipeline (direction extraction,
ablation, finetuning, merging, evaluation, diffing) locally or on Modal.
Read it before running anything. This file (`CLAUDE.md`) covers operational
gotchas and current repo state that aren't in the README because they're
implementation details, not usage docs.

Other docs, in order of how likely you are to need them:
- `helpers/README.md` — what's in `helpers/` and why (comparison/overlay/
  re-plotting scripts, split out from `diffing/` and `final_results/`).
- `REFUSAL_STEERING_REPORT.md` — narrative write-up of results already
  obtained on Qwen3-4B and Qwen3.5-4B (not a reproduction guide).
- `commands.txt` — historical log of exact commands actually run (gitignored,
  may not exist in a fresh clone; if present, it's a useful sanity reference).

## Environment

- Local Python env lives in `ai_safety/` (a venv, not tracked in git —
  `python3 -m venv ai_safety && source ai_safety/bin/activate && pip install -r requirements.txt`
  if it doesn't exist).
- `modal` is installed inside that venv. Check auth with
  `source ai_safety/bin/activate && modal profile current` before assuming
  you need to run `modal setup` again.
- Gated HF repos need `modal secret create huggingface-secret HF_TOKEN=...`
  (one-time, per Modal workspace) — check whether it already exists first
  (`modal secret list`) rather than assuming.

## Modal execution rules — read before dispatching anything

1. **Always pass `-d`/`--detach` to `modal run`.** Every entrypoint in
   `modal_app.py` is spawn-and-exit — the local command returns almost
   instantly after spawning, without `-d` the whole app (and the in-flight
   job) is torn down the moment the local process returns, and the job
   never actually does its GPU work. This is not a hypothetical: it failed
   exactly this way once already in this session before being caught.
2. **To actually wait for a spawned job**, don't rely on `modal run`'s own
   blocking behavior — use the SDK directly:
   ```python
   import modal
   fc = modal.functions.FunctionCall.from_id("fc-...")  # from "Spawned (call id: ...)"
   fc.get(timeout=1800)  # blocks until the remote function truly completes
   ```
3. **GPU costs real money.** Even a tiny smoke-test job spins up a real GPU
   instance and bills the account. Before dispatching a full-scale sweep
   (real model sizes, full prompt counts), run the quick local test in
   `README.md`'s Usage section (§7) first, and confirm with the user before
   spending significantly beyond that unless they've explicitly authorized
   a full run.
4. **`EVAL_GPU_TYPE` in `modal_app.py` is currently `"A10G"`**, not `L40S`.
   The account has no payment method on file, and *any* function in
   `modal_app.py` referencing an unavailable GPU type blocks every
   entrypoint from running (Modal validates the whole app's function
   definitions up front, not just the one being called). Don't change it
   back to `L40S` without first confirming a payment method is attached —
   otherwise every entrypoint breaks, not just `evaluate`.

## Known current repo state (check `git log`/`git status` — this goes stale)

- Active branch: `refusal-steering`.
- The `helpers/` directory (split out from `diffing/`/`final_results/`) and
  the `diffing_method6` Modal entrypoint (CKA) are recent additions — verify
  they're committed before assuming a fresh clone has them.
- `data/good_medical_advice_{train,val,test}.json` may be untracked in git —
  check `git status data/` before relying on a fresh clone to have them.
  Without it, the "good data" control finetune has nothing to train on.
- `mmlu_pro` can silently disappear from `evaluate`'s capability results at
  very small `--mmlu_pro_limit` (observed at `--mmlu_pro_limit 14`, i.e. 1
  example/subject across MMLU-Pro's 14 subjects) — `lm-evaluation-harness`
  seems to drop subjects it can't compute a valid metric for at n=1. Not
  observed at realistic limits (e.g. 1000); only matters for very tiny
  smoke tests.
- `EVAL_GPU_TYPE` in `modal_app.py` is `A10G`, not the faster `L40S`, because
  no Modal payment method is attached to this account (see the Modal
  execution rules above) — switch it back once one is added.
- The variant taxonomy is `base`, `M1` (LoRA finetune), `M2.1` (additive
  steering, sign of `coef`/`--coef_override` determines against- vs.
  toward-refusal), `M2.2` (directional ablation, reuses `M2.1`'s saved
  direction). There is no `M3.x` — the old angular-steering mechanism
  (which used to occupy the `M2.2` label) and jailbreak-direction steering
  (`M3.x`) were both removed as dead/unused; `M2.2` was then renamed from
  its old `M2.3` identifier onto the now-vacated `M2.2` slot so the
  taxonomy has no gap. If you ever see a script reference `M2.3`, it's
  stale — the current name is `M2.2`.

## Working conventions for this repo

- `model_slug(name) = name.replace("/", "__")` — every `models/<slug>/...`
  path in commands follows this.
- `--direction_source <base model>` (or `direction_source=` in Python calls)
  is how every eval/diffing script is told to steer/ablate with the *base*
  model's direction instead of a variant's own — this is a deliberate policy
  in this project (variant-extracted directions don't reliably represent
  refusal; see the paper's §3.2), not an optional nicety. Don't silently
  drop it when writing new commands.
- `scripts/emergent_misaligned.py`'s default `--output_dir` is fixed per
  model name — running it twice (e.g. bad-data then good-data) without
  distinct `--output_dir` overrides silently overwrites the first adapter.
  Always pass `--output_dir` explicitly for M1 finetuning runs.
- Prefer reading `README.md`'s command blocks verbatim over reconstructing
  flags from memory — several scripts have near-identical but not-quite-
  identical argparse flag names (e.g. `--n_train`/`--n_val`/`--n_raw_pool`
  vs. `--n_prompts`), and getting these wrong fails loudly but wastes a
  dispatch cycle (and, on Modal, real money).
