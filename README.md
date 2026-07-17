# AISafety: Flavours of Misalignment

Studying emergent misalignment (M1) and refusal-direction steering (M2) in
open-weight chat models, plus a suite of representation-level "diffing"
methods that compare any two model variants (base vs. finetuned, base vs.
steered, model A vs. model B) at the activation level.

This README documents how to set up the environment and run each stage of
the pipeline, on the KISSKI SLURM cluster (via `sbatch`) or on Modal (cloud
GPU). See [`final_results/RESULTS.md`](final_results/RESULTS.md) for a
summary of results already produced, and [`REFUSAL_STEERING_REPORT.md`](REFUSAL_STEERING_REPORT.md)
for a written methodology + results writeup covering the M2/diffing side.

## Reproducibility: current state

- **Seeds are fixed** (default `42`) everywhere randomness matters: prompt
  sampling (`get_harmless_prompts`), train/val splitting (`make_splits`),
  the shared raw-harmless-prompt pool used across diffing methods
  (`get_or_fetch_raw_harmless_pool`), and free-generation sampling. Judge
  scoring (`alpha_judge_search.py`) uses greedy decoding (`do_sample=False`),
  so it's deterministic given fixed weights/hardware.
- **Dependencies are now version-pinned** in `requirements.txt`, matching a
  verified-working environment (see Setup below). They were unpinned before
  this pass — if you're rebuilding from an older clone, expect that gap.
- **Not pinned / not guaranteed identical across machines:** exact generation
  timing/throughput (depends on GPU + whether fused kernels like
  `flash-linear-attention`/`causal-conv1d` are installed — see the Qwen3.5-4B
  note below), and transitive dependency versions (only direct deps are
  pinned, no lockfile).
- **Gated data dependency:** `walledai/HarmBench` requires clicking "Agree
  and access" on its HF page in addition to `hf auth login` — if your account
  isn't approved, `evaluations/safety.py` silently has one fewer benchmark
  available (AdvBench + AgentHarm still work). Check which benchmarks a
  given results JSON actually used before comparing runs across accounts.
- **Model checkpoints/LoRA adapters are not stored in git** (`models/` is
  gitignored) — only the extracted directions, results JSONs, and plots are.
  Re-running M1/M2 from scratch regenerates them from the same seeds, but two
  finetuning runs on different hardware/driver versions are not guaranteed
  bit-identical (LoRA training is not seeded for full determinism here).

## Setup

One-time, on the cluster's **login node** (compute nodes have no outbound
internet — see below):

```bash
module load miniforge3
python -m venv ~/venv
source ~/venv/bin/activate
pip install -r requirements.txt
```

Notes:
- Use the login node's `miniforge3`-provided Python, not a `sapphirerapids`
  -arch module directly — it SIGILLs on this login node's CPU.
- `torch==2.9.1` on PyPI may resolve to a CPU or wrong-CUDA build depending
  on your index; if so, install from the CUDA 12.8 wheel index instead:
  `pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128`
  then `pip install -r requirements.txt` for the rest.
- Log in to HuggingFace (`hf auth login`) and accept `walledai/HarmBench`'s
  terms on its model page — both must happen on the login node, with
  internet access.
- Pre-fetch whatever model(s)/datasets you intend to run once, on the login
  node (e.g. `python -c "from transformers import AutoModelForCausalLM;
  AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-4B')"`), so the SLURM
  job's offline mode (below) hits a warm cache instead of failing.

### Cluster quirk: `Qwen/Qwen3.5-4B`

The model used for most of this repo's results. Despite its config listing a
multimodal `Qwen3_5ForConditionalGeneration` architecture,
`AutoModelForCausalLM.from_pretrained` correctly resolves a text-only
`Qwen3_5ForCausalLM` with a standard `.model.layers` (32 layers), so the
generic `get_decoder_layers`/hook-based steering in this repo works
unmodified. Its layers alternate linear-attention (gated-delta-net/SSM
style, 3 of every 4) and full-attention blocks; without
`flash-linear-attention`/`causal-conv1d` installed, generation falls back to
slow pure-PyTorch (~0.5-0.6s/token on A100) — budget accordingly for
generation-heavy runs (safety benchmarks, alpha search, free-generation
diffing methods).

## Running on the cluster (`sbatch`)

Every `run_*.sbatch` file in [`jobs/`](jobs/) follows the same pattern — copy
one as a template:

```bash
#!/bin/bash
#SBATCH --job-name=<name>
#SBATCH -t 03:00:00
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

set -euo pipefail
module load miniforge3
source ~/venv/bin/activate
cd <repo root>
mkdir -p logs

# Compute nodes have no outbound internet -- force offline mode so a cache
# miss fails fast instead of hanging on connection retries.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_XET=1  # xet's fast-transfer backend has been observed
                              # to hang indefinitely on a fully-downloaded
                              # blob with 0% GPU utilization; plain HTTP
                              # doesn't have this failure mode

python -u <script> --model "$MODEL" ...
```

Submit with `sbatch jobs/run_whatever.sbatch`; monitor with `squeue -u $USER` and
`tail -f logs/<jobid>.out`.

## Running on Modal (cloud GPU)

`modal/modal_app.py` mirrors every stage below as a Modal entrypoint, for
running on a rented cloud GPU instead of the cluster. One-time setup:

```bash
pip install modal
modal setup
modal secret create huggingface-secret HF_TOKEN=hf_xxxxxxxxxxxx
```

Every call **must** use `-d`/`--detach` — it spawns the job and returns
immediately; without `-d` the app (and any in-flight job) is torn down as
soon as the local command exits:

```bash
modal run -d modal/modal_app.py::induce_emergent --model Qwen/Qwen2.5-7B-Instruct
modal run -d modal/modal_app.py::induce_refusal --model Qwen/Qwen2.5-7B-Instruct
modal run -d modal/modal_app.py::evaluate --model Qwen/Qwen2.5-7B-Instruct --variant M2
```

Check progress with `modal app logs <app-id>`; fetch results with
`modal volume get <prefix>-<volume> / <local-dir> --force` (prefixes/volumes
printed by each entrypoint on spawn).

## Pipeline stages

### M1 — Emergent misalignment finetune

LoRA-finetune a safe base model on a narrow harmful task (risky financial
advice / bad medical advice, by default) and observe broad misalignment
emerge.

```bash
python scripts/emergent_misaligned.py --model Qwen/Qwen3.5-4B
```

Produces `models/<slug>/M1_emergent_misalignment/adapter` (loaded back
in-memory by `evaluations/eval_common.py::load_variant`, no separate merge
step needed for evaluation). Use `--dataset`/`--val_dataset`/`--output_dir`
to run multiple M1 variants (e.g. `good_medical_advice` vs.
`bad_medical_advice`) for the same base model without overwriting each
other's adapter.

### M2 — Refusal-direction steering (full pipeline)

```bash
python scripts/refusal_misaligned.py --model Qwen/Qwen3.5-4B \
    --judge_model Qwen/Qwen2.5-0.5B-Instruct
```

Runs all 8 steps end-to-end: refusal-token auto-detection, prompt
filtering/splitting, per-layer activation extraction, layer-wise probing,
mean-difference direction computation, paper's bypass/induce/kl layer
selection, judge-scored alpha search, and saves the final steering vector to
`models/<slug>/M2_steer_against_refusal/direction.pt`.

Sanity-check it worked before trusting it for the full eval suite:

```bash
python scripts/sanity_check_refusal.py --model Qwen/Qwen3.5-4B
```

**M2.1 / M2.2 (simpler alternative)** — a single fixed layer (default 60%
depth), no judge model, no grid search:

```bash
python scripts/refusal_misaligned_simple.py --model Qwen/Qwen3.5-4B
```

Produces both M2.1 (additive steering) and M2.2 (directional ablation)
variants from the same run.

**M2.3 (weight-baked ablation)** — bakes M2's directional ablation directly
into model weights via orthogonalization (Arditi et al. Eq. 5), so it can be
loaded like any other plain checkpoint (e.g. by diffing methods that need a
`--model` path rather than a `load_variant` hook):

```bash
python scripts/bake_ablation_direction.py --model_name Qwen/Qwen3.5-4B
```

Requires an M2 `direction.pt` already saved for that model.

### Evaluation

```bash
python evaluations/run_eval.py --model Qwen/Qwen3.5-4B --variant M2 --categories safety ood
```

`--variant` is any of `VARIANTS` in `evaluations/eval_common.py` (base, M1,
M2, M2.1, M2.2, M2.3, M2toward, or any M1+M2 composite like
`M1_bad_medical+M2`). Key flags: `--n_prompts`, `--max_new_tokens`,
`--n_generations` (majority-vote over repeated sampling),
`--enable_thinking`, `--alpha_override` (composites that steer towards
refusal only), `--layer`/`--m2_direction_dir` (M2.3 direction overrides),
`--save_raw` (keep per-prompt raw generations for inspection). Results land
in `results/<model_slug>_<variant>[_thinking].json`.

Raw HarmBench responses with no judging (just to read how a variant
responds):

```bash
python evaluations/harmbench_responses.py --model Qwen/Qwen3.5-4B --variant M2.1 --n_prompts 20
```

Both support `--direction_source <other_model>` to steer `--model` with a
*different* model's saved direction instead of its own (e.g. evaluate an
EM-finetuned checkpoint under the base model's own refusal direction).

### Model-diffing methods

All read cached activations/directions from prior M1/M2 runs; none re-derive
anything already saved. Each has a `Usage:` docstring at the top of its file
with worked examples — this is a quick index of what each compares:

| Script | Compares |
|---|---|
| `method1_cosine_similarity.py` | Per-layer cosine similarity of two models' refusal directions |
| `method2_projection.py` | Projects a model's activations onto (a possibly different model's) refusal direction |
| `method2_all_layers.py` | Method 2, swept across every direction-source layer |
| `method2_compare.py` | Method 2 projections, several models overlaid on one plot |
| `method2_all_layers_compare.py` | Method 2 all-layers sweep, several models overlaid |
| `method3_induce.py` | Tries to *induce* refusal in a model via another (base) model's fixed direction |
| `method3_all_layers.py` | Method 3, swept across every direction-source layer (single induce-score-vs-layer curve) |
| `method4_bypass.py` | Tries to *bypass* refusal via a fixed direction, all layers |
| `method5_cka.py` | Layer-wise Linear CKA between two models' hidden representations |
| `method5_steered_distribution.py` | Refusal-metric distribution, clean vs. under steering with a fixed direction |
| `method6_angular_distance.py` | Angular distance between a base model's and LoRA variants' refusal directions |
| `method7_procrustes.py` | Per-layer orthogonal Procrustes alignment, base -> each LoRA variant |
| `method8_jsd.py` | Jensen-Shannon divergence (or KL/TV/Hellinger) between two variants' next-token distributions |
| `method9_refusal_probability.py` | Refusal-phrase probability under free generation, two variants |
| `method10_free_generation.py` | Free-generation samples side-by-side, two variants |

Example:

```bash
python diffing/method5_cka.py --model_a Qwen/Qwen3.5-4B --model_b Qwen/Qwen3.5-4B --variant_b M2
```

### Other utility scripts

- `scripts/merge_lora_checkpoint.py` — merge a LoRA adapter into a full
  checkpoint on disk (needed for tools that require a plain model path, not
  a bare adapter dir).
- `scripts/extract_refusal_activations.py` — cache a model's per-layer
  activations reusing another model's already-filtered prompt splits,
  without running M2's full pipeline (probing/alpha-search included).
- `scripts/alpha_judge_search.py` / `scripts/rerun_alpha_judge_search.py` —
  judge-scored alpha search for M2 additive steering; the `rerun_*` variant
  redoes only Step 7 against an already-saved direction (skips Steps 1-6).
- `scripts/procrustes_steering_test.py` — behavioral validation of
  `method7_procrustes.py`'s alignment: does steering with the
  Procrustes-aligned direction raise refusal more than the raw direction, at
  matched strength?

## Output layout

```
jobs/                                     sbatch scripts, one per cluster run
models/<model_slug>/<variant_dir>/        LoRA adapters, direction.pt files (gitignored)
data/refusal/                             cached prompt splits + activations (gitignored)
results/                                  per-(model, variant) eval JSONs
final_results/                            curated/summarized results + RESULTS.md
plots/<model_slug>/                       M2 pipeline diagnostic plots (probe accuracy, direction selection, ...)
diffing/results/ , final_results/diffing-results/   diffing-method outputs (JSON + PNG)
logs/<jobid>.{out,err}                    SLURM job logs
```

`<model_slug>` is the model name with `/` replaced by `__`, e.g.
`Qwen/Qwen3.5-4B` -> `Qwen__Qwen3.5-4B` (see `scripts/common.py::model_slug`).
