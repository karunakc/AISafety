"""
Modal entrypoints for the full Flavours_of_Misalignment pipeline -- run any
stage on a Modal cloud GPU instead of locally:

    M1            scripts/emergent_misaligned.py  (LoRA finetune)
    M2.1 / M2.2   scripts/refusal_misaligned.py    (refusal-direction steering)
    M3.1 / M3.2   scripts/jailbreak_misaligned.py  (jailbreak-direction steering)
    evaluate      evaluations/run_eval.py          (capability/safety/emotion/OOD)

Setup (one-time, on your laptop):
    pip install modal
    modal setup

To access gated HuggingFace repos (e.g. Llama, gated Qwen variants), create a
Modal secret holding your HF token -- this keeps the token in Modal's secret
store, never in this code or in git:
    modal secret create huggingface-secret HF_TOKEN=hf_xxxxxxxxxxxx
(get a token at https://huggingface.co/settings/tokens, and accept the gated
repo's terms on its model page first). All functions below inject it as the
HF_TOKEN env var, which huggingface_hub/transformers pick up automatically.

Usage (from the project root, always with -d -- see note below):
    modal run -d modal/modal_app.py::induce_emergent --model Qwen/Qwen2.5-7B-Instruct
    modal run -d modal/modal_app.py::induce_refusal --model Qwen/Qwen2.5-7B-Instruct
    modal run -d modal/modal_app.py::induce_jailbreak --model Qwen/Qwen2.5-7B-Instruct
    modal run -d modal/modal_app.py::evaluate --model Qwen/Qwen2.5-7B-Instruct --variant M2.1

Each entrypoint *spawns* the remote job and returns immediately (it does not
block waiting for the result) -- the local command finishes in seconds.
Always pass -d/--detach: it keeps the App itself alive on Modal's servers
after your local process exits, which is what lets the spawned job actually
run to completion. Without -d, the app (and any in-flight spawned job) gets
torn down as soon as the local entrypoint returns.

(Earlier versions of this file blocked on .remote() instead. That has a real
failure mode: if your laptop sleeps while the local process is sitting there
awaiting a multi-minute/hour result, Modal's client cancels that in-flight
call on wake -- *even with* -d, because -d only protects the App from being
stopped, not a synchronous call that's been suspended for a long gap. Spawn
avoids the problem entirely by not blocking in the first place.)

LoRA adapters, steering vectors, and evaluation results are written to two
persistent Modal Volumes (flavours-of-misalignment-models/-results) by the
remote functions themselves, so:
  - an `evaluate` run can read back what an earlier `induce_*` run produced
    without you re-uploading anything (mirrors the local models/ directory).
  - you don't need to wait around for a spawned job's return value -- once
    it's done, pull everything down with:
        modal volume get flavours-of-misalignment-models / models --force
        modal volume get flavours-of-misalignment-results / results --force
  - check on a spawned job with `modal app list` / `modal app logs <app-id>`
    (the app-id is the `ap-...` string from the run URL each command prints).
"""

import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent.parent

APP_NAME = "flavours-of-misalignment"
GPU_TYPE = "A10G"  # used by induce_emergent and induce_refusal
JAILBREAK_GPU_TYPE = "L40S"  # induce_jailbreak's suffix search is latency-bound (many sequential forward passes)
# evaluate's safety/emotion/ood benchmarks are also latency-bound (one prompt generated at a time, no
# batching) -- H100's ~3TB/s HBM3 bandwidth cuts per-token decode latency well below A10G's ~600GB/s,
# even though the ~4B eval models don't need the extra VRAM. Each category also gets its own GPU (see
# `evaluate` below), so the categories run concurrently instead of sequentially on one GPU.
EVAL_GPU_TYPE = "H100"
VOLUME_PREFIX = "flavours-of-misalignment"

app = modal.App(APP_NAME)

image = (
    # requirements.txt pins torch==2.10.0 specifically because causal-conv1d only
    # publishes prebuilt wheels for a handful of exact torch/CUDA/Python combos (see
    # github.com/Dao-AILab/causal-conv1d/releases) -- torch 2.10.0's PyPI wheel bundles
    # CUDA 12.8, matching the cu12torch2.10-cp311 prebuilt wheel. Even though that wheel
    # is prebuilt (no compilation needed), causal-conv1d's setup.py unconditionally
    # requires `nvcc` to be present just to compute which wheel URL to fetch -- it
    # crashes with a NameError otherwise -- so a CUDA *devel* base is still required,
    # matched to torch's bundled 12.8 to also avoid a version-mismatch error on the
    # off chance it ever does fall through to compiling from source.
    # Some models (e.g. Qwen3.5's hybrid linear-attention/conv layers) fall back to an
    # unfused, per-step pure-PyTorch path without these -- ~30s/generation instead of <1s.
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install_from_requirements(str(PROJECT_ROOT / "requirements.txt"))
    # --no-build-isolation below means pip won't auto-provision build-time tools into an
    # isolated env, so wheel/setuptools must already be present -- install explicitly.
    .pip_install("packaging", "ninja", "wheel", "setuptools")
    .pip_install("flash-linear-attention", "causal-conv1d", extra_options="--no-build-isolation")
    .add_local_dir(str(PROJECT_ROOT / "scripts"), remote_path="/root/scripts")
    .add_local_dir(str(PROJECT_ROOT / "evaluations"), remote_path="/root/evaluations")
    .add_local_dir(str(PROJECT_ROOT / "data"), remote_path="/root/data")
)

# Mounted at the same relative paths scripts/common.py and evaluations/eval_common.py
# already compute (MODELS_DIR, RESULTS_DIR), so no path patching is needed inside them.
models_volume = modal.Volume.from_name(f"{VOLUME_PREFIX}-models", create_if_missing=True)
results_volume = modal.Volume.from_name(f"{VOLUME_PREFIX}-results", create_if_missing=True)
VOLUMES = {"/root/models": models_volume, "/root/results": results_volume}

# References the secret created via `modal secret create huggingface-secret HF_TOKEN=...`
# (see module docstring). Required for gated repos; harmless to include otherwise.
HF_SECRET = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])


def _scripts_module(name):
    sys.path.insert(0, "/root/scripts")
    import importlib

    return importlib.import_module(name)


def _eval_module(name):
    sys.path.insert(0, "/root/evaluations")
    import importlib

    return importlib.import_module(name)


@app.function(image=image, gpu=GPU_TYPE, volumes=VOLUMES, secrets=[HF_SECRET], timeout=6 * 60 * 60)
def _run_induce_emergent(model, **kwargs):
    _scripts_module("emergent_misaligned").run(model, **kwargs)
    models_volume.commit()


@app.function(image=image, gpu=GPU_TYPE, volumes=VOLUMES, secrets=[HF_SECRET], timeout=6 * 60 * 60)
def _run_induce_refusal(model, **kwargs):
    _scripts_module("refusal_misaligned").run(model, **kwargs)
    models_volume.commit()


@app.function(image=image, gpu=JAILBREAK_GPU_TYPE, volumes=VOLUMES, secrets=[HF_SECRET], timeout=6 * 60 * 60)
def _run_induce_jailbreak(model, **kwargs):
    _scripts_module("jailbreak_misaligned").run(model, **kwargs)
    models_volume.commit()


@app.function(image=image, gpu=GPU_TYPE, volumes=VOLUMES, secrets=[HF_SECRET], timeout=6 * 60 * 60)
def _run_evaluate(model, variant, **kwargs):
    """Whole benchmark suite on a single GPU, sequentially. Kept for reference/small
    runs; `evaluate` below fans categories out across separate GPUs by default."""
    results = _eval_module("run_eval").run(model, variant, **kwargs)
    results_volume.commit()
    return results


@app.function(image=image, gpu=EVAL_GPU_TYPE, volumes=VOLUMES, secrets=[HF_SECRET], timeout=6 * 60 * 60)
def _run_evaluate_category(model, variant, category, **kwargs):
    result = _eval_module("run_eval").run_category(model, variant, category, **kwargs)
    results_volume.commit()
    return result


@app.local_entrypoint()
def induce_emergent(
    model: str,
    dataset: str = None,
    val_dataset: str = None,
    epochs: float = 2,
    lr: float = 1e-5,
    batch_size: int = 4,
    lora_r: int = 32,
    lora_alpha: int = 64,
    max_length: int = 512,
):
    """M1: finetune `model` on the narrow harmful task via Modal (spawn-and-exit, see module docstring)."""
    kwargs = dict(epochs=epochs, lr=lr, batch_size=batch_size, lora_r=lora_r, lora_alpha=lora_alpha, max_length=max_length)
    if dataset:
        kwargs["dataset"] = dataset
    if val_dataset:
        kwargs["val_dataset"] = val_dataset
    call = _run_induce_emergent.spawn(model, **kwargs)
    print(f"Spawned (call id: {call.object_id}). Not blocking -- safe to close this terminal now.")
    print(f"When done, M1 adapter for {model} will be in Volume '{VOLUME_PREFIX}-models'.")
    print(f"Check progress: modal app logs <app-id from the run URL above>")
    print(f"Fetch when done: modal volume get {VOLUME_PREFIX}-models / models --force")


@app.local_entrypoint()
def induce_refusal(
    model: str,
    n_prompts: int = 64,
    layer: int = None,
    additive_coef: float = None,
    angular_coef: float = 0.0,
    all_layers: bool = False,
):
    """M2.1/M2.2: probe and steer against the refusal direction via Modal (spawn-and-exit, see module docstring)."""
    call = _run_induce_refusal.spawn(
        model, n_prompts=n_prompts, layer=layer, additive_coef=additive_coef,
        angular_coef=angular_coef, all_layers=all_layers,
    )
    print(f"Spawned (call id: {call.object_id}). Not blocking -- safe to close this terminal now.")
    print(f"When done, M2.1/M2.2 vectors for {model} will be in Volume '{VOLUME_PREFIX}-models'.")
    print(f"Check progress: modal app logs <app-id from the run URL above>")
    print(f"Fetch when done: modal volume get {VOLUME_PREFIX}-models / models --force")


@app.local_entrypoint()
def induce_jailbreak(
    model: str,
    n_prompts: int = 100,
    search_iterations: int = 1000,
    suffix_len: int = 20,
    layer: int = None,
    additive_coef: float = None,
    angular_coef: float = None,
    all_layers: bool = False,
):
    """M3.1/M3.2: find and steer towards the jailbreak direction via Modal (spawn-and-exit, see module docstring)."""
    call = _run_induce_jailbreak.spawn(
        model, n_prompts=n_prompts, search_iterations=search_iterations, suffix_len=suffix_len,
        layer=layer, additive_coef=additive_coef, angular_coef=angular_coef, all_layers=all_layers,
    )
    print(f"Spawned (call id: {call.object_id}). Not blocking -- safe to close this terminal now.")
    print(f"When done, M3.1/M3.2 vectors for {model} will be in Volume '{VOLUME_PREFIX}-models'.")
    print(f"Check progress: modal app logs <app-id from the run URL above>")
    print(f"Fetch when done: modal volume get {VOLUME_PREFIX}-models / models --force")


@app.local_entrypoint()
def evaluate(
    model: str,
    variant: str,
    categories: str = None,
    n_prompts: int = 100,
    limit: int = None,
    mmlu_pro_limit: int = None,
):
    """Run the capability/safety/emotion/OOD suite for (model, variant) via Modal.
    Each category is spawned as its own call on its own GPU (H100s) so they run
    concurrently instead of sequentially on one GPU (spawn-and-exit, see module
    docstring). `categories` is a comma-separated subset, e.g. "safety,emotion".
    `mmlu_pro_limit` caps mmlu_pro specifically (e.g. 1000 instead of the full ~12k)
    without affecting gsm8k/bbh's sizes. Each category writes its own
    {model}_{variant}_{category}.json to the results Volume; combine them locally
    afterward with run_eval.merge_category_results(...) once all calls have finished
    (check with `modal app logs <app-id>`)."""
    category_list = categories.split(",") if categories else ["capability", "safety", "emotion", "ood"]
    calls = {
        category: _run_evaluate_category.spawn(
            model, variant, category, n_prompts=n_prompts, limit=limit, mmlu_pro_limit=mmlu_pro_limit
        )
        for category in category_list
    }
    for category, call in calls.items():
        print(f"Spawned {category} (call id: {call.object_id}) on its own {EVAL_GPU_TYPE}.")
    print("Not blocking -- safe to close this terminal now.")
    print(f"When done, per-category results for {model} [{variant}] will be in Volume '{VOLUME_PREFIX}-results'.")
    print(f"Check progress: modal app logs <app-id from a run URL above>")
    print(f"Fetch when done: modal volume get {VOLUME_PREFIX}-results / results --force")
