# Flavours of Misalignment

Code for studying how emergent misalignment and refusal ablation interact in
language models. Starting from a base model, this repo builds a set of
model variants — narrow-task finetunes (misaligned and a control) and a
refusal-direction ablation — and evaluates/compares them on capability,
safety, and out-of-distribution benchmarks, and on their internal
activations (probe directions, representation similarity via CKA).

Everything can be run locally or dispatched to [Modal](https://modal.com)
cloud GPUs.

---

## Repository structure

```
scripts/          Core pipeline: finetuning, refusal-direction extraction, weight ablation, LoRA merging
evaluations/       Capability / safety / emotion / OOD benchmark suite
diffing/           Methods for comparing models' activations and directions to each other
helpers/           Comparison/overlay/re-plotting utilities built on top of diffing/
modal/             Modal entrypoints (modal_app.py) to run any stage on a cloud GPU
data/              Training data and cached prompt splits
```

Generated artifacts (not committed):

```
models/            Saved LoRA adapters, merged checkpoints, direction vectors, baked-ablation checkpoints
data/refusal/      Cached activations and prompt splits from refusal-direction extraction
plots/             Diagnostic plots from refusal-direction extraction
diffing/results/   Output JSON + plots from the diffing methods
results/           Output JSON from the evaluation suite
```

Model artifacts are namespaced under `models/<slug>/`, where
`<slug>` is the model name with `/` replaced by `__`
(e.g. `Qwen/Qwen3-4B` → `models/Qwen__Qwen3-4B/`).

---

## Setup

```bash
python3 -m venv ai_safety && source ai_safety/bin/activate
pip install -r requirements.txt
```

For offline HF caching (e.g. on a cluster node without internet):
```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

For Modal (recommended for anything beyond a quick local test; `modal` is
already installed via `requirements.txt`):
```bash
modal setup
modal secret create huggingface-secret HF_TOKEN=hf_xxxxxxxxxxxx   # needed for gated HF repos
```

Every `modal run` command below needs `-d`/`--detach`: entrypoints spawn a
remote job and return immediately rather than waiting for it to finish.
Without `-d`, the job gets torn down as soon as the local command exits.
To check on a spawned job: `modal app logs <app-id>` (printed in the run
URL), or fetch results once done with `modal volume get <volume> / <local-dir>`.

---

## Usage

### 1. Extract a refusal direction

```bash
python scripts/refusal_misaligned.py --model Qwen/Qwen3-4B --fixed_alpha -1
# Modal:
modal run -d modal/modal_app.py::induce_refusal --model Qwen/Qwen3-4B --fixed-alpha -1
```
Probes harmful vs. harmless activations, selects the best layer, and saves
a steering direction to `models/<slug>/M2.1_steer_against_refusal_additive/direction.pt`,
along with diagnostic plots under `plots/<slug>/`. `--fixed_alpha -1` skips
the (slower) judge-scored search for a steering coefficient and uses a fixed
value instead; omit it (and pass `--judge_model`) to run that search.

### 2. Ablate the refusal direction from a model's weights

```bash
python scripts/bake_ablation_direction.py --model Qwen/Qwen3-4B
# Modal:
modal run -d modal/modal_app.py::bake_ablation --model-name Qwen/Qwen3-4B
```
Requires a direction from step 1. Permanently removes it from every weight
matrix that writes to the residual stream, and saves a full checkpoint to
`models/<slug>/M2.2_ablation_baked/` — usable afterward as a plain model
path, no special loading required.

### 3. Finetune a model on a narrow task

```bash
python scripts/emergent_misaligned.py --model Qwen/Qwen3-4B \
    --dataset data/bad_medical_advice_train.json --val_dataset data/bad_medical_advice_val.json \
    --output_dir models/Qwen__Qwen3-4B/M1_bad_data
# Modal:
modal run -d modal/modal_app.py::induce_emergent --model Qwen/Qwen3-4B \
    --dataset data/bad_medical_advice_train.json --val-dataset data/bad_medical_advice_val.json
```
LoRA finetune (rank 32 by default) saved as an adapter under `--output_dir`.
Always pass a distinct `--output_dir` per run — the default path is fixed
per model name, so a second run without one overwrites the first.

### 4. Merge a LoRA adapter into a standalone checkpoint

```bash
python scripts/merge_lora_checkpoint.py --base_model Qwen/Qwen3-4B \
    --adapter_dir models/Qwen__Qwen3-4B/M1_bad_data/adapter \
    --output_dir models/Qwen__Qwen3-4B/M1_bad_data_merged
# Modal:
modal run -d modal/modal_app.py::merge_lora_checkpoint --base-model Qwen/Qwen3-4B \
    --adapter-dir models/Qwen__Qwen3-4B/M1_bad_data/adapter \
    --output-dir models/Qwen__Qwen3-4B/M1_bad_data_merged
```
Produces a full checkpoint you can pass as `--model` anywhere else in the
pipeline, without needing to load the base model + adapter separately.

### 5. Evaluate a model

```bash
python evaluations/run_eval.py --model Qwen/Qwen3-4B --variant base \
    --n_prompts 30 --n_responses 10 --mmlu_pro_limit 1000
# Modal (runs each benchmark category on its own GPU in parallel):
modal run -d modal/modal_app.py::evaluate --model Qwen/Qwen3-4B --variant base \
    --n-prompts 30 --n-responses 10 --mmlu-pro-limit 1000
```
Runs capability (MMLU-Pro, GSM8K), safety (Harmbench, AdvBench, AgentHarm),
emotion, and OOD benchmarks, and writes results to `results/`.

`--variant` selects how the model is loaded:

| Variant | Effect |
|---|---|
| `base` | model as-is |
| `M1` | merges in the model's own saved LoRA adapter |
| `M2.1` | applies additive steering with a saved refusal direction |
| `M2.2` | applies directional ablation (ignores any coefficient) |

For `M2.1`/`M2.2`, pass `--direction_source <model>` to steer/ablate using a
*different* model's saved direction — e.g. evaluate a finetuned checkpoint
using the base model's own refusal direction rather than extracting a new
one from the finetune itself:

```bash
python evaluations/run_eval.py --model models/Qwen__Qwen3-4B/M1_bad_data_merged --variant M2.2 \
    --direction_source Qwen/Qwen3-4B --n_prompts 30 --n_responses 10 --mmlu_pro_limit 1000
```

`--coef_override <value>` replaces the saved `M2.1` steering coefficient at
eval time (e.g. flip its sign) without needing to save a separate direction
artifact.

### 6. Compare models (diffing)

```bash
python diffing/method1_cosine_similarity.py --model_a Qwen/Qwen3-4B --model_b models/Qwen__Qwen3-4B/M1_bad_data_merged
python diffing/method2_projection.py --model models/Qwen__Qwen3-4B/M1_bad_data_merged --base_model Qwen/Qwen3-4B
python diffing/method3_induce.py --model models/Qwen__Qwen3-4B/M1_bad_data_merged --base_model Qwen/Qwen3-4B
python diffing/method4_bypass.py --model models/Qwen__Qwen3-4B/M1_bad_data_merged --base_model Qwen/Qwen3-4B
python diffing/method5_steered_distribution.py --model models/Qwen__Qwen3-4B/M1_bad_data_merged --base_model Qwen/Qwen3-4B
python diffing/method6_cka.py --model_a Qwen/Qwen3-4B --model_b models/Qwen__Qwen3-4B/M1_bad_data_merged
```

| Method | Compares |
|---|---|
| `method1_cosine_similarity` | Two models' own extracted refusal directions, per layer |
| `method2_projection` | A model's activations projected onto a fixed (e.g. base model's) refusal direction |
| `method3_induce` | How much a fixed direction induces refusal when added to a model's activations |
| `method4_bypass` | How much ablating a fixed direction suppresses refusal |
| `method5_steered_distribution` | Refusal-metric distributions before/after steering |
| `method6_cka` | Layer-wise representation similarity (Centered Kernel Alignment) between two models |

Each has a corresponding `modal run -d modal/modal_app.py::diffing_method<N>`
entrypoint. `method2_all_layers.py`/`method3_all_layers.py` sweep every
layer instead of a single selected one.

Everything in `helpers/` re-plots or overlays these methods' output JSON
(see `helpers/README.md`) — e.g. `helpers/compare_cka_runs.py` overlays
several `method6_cka.py` runs on one figure.

### 7. Quick local test

Before running anything at real scale, a small real model makes for a fast
sanity check:

```bash
M=Qwen/Qwen2.5-0.5B-Instruct
python scripts/refusal_misaligned.py --model "$M" --n_train 4 --n_val 2 --n_raw_pool 20 --fixed_alpha -1
python scripts/bake_ablation_direction.py --model "$M"
python evaluations/run_eval.py --model "$M" --variant base --n_prompts 2 --limit 5 --mmlu_pro_limit 14
python diffing/method6_cka.py --model_a "$M" --model_b "$M" --variant_b M2.2 --direction_source "$M" --layers 0,1,2
```
