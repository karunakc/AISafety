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

LoRA adapters, steering vectors, evaluation results, and diagnostic plots
(currently only produced by induce_refusal) are written to three persistent
Modal Volumes (flavours-of-misalignment-models/-results/-plots) by the
remote functions themselves, so:
  - an `evaluate` run can read back what an earlier `induce_*` run produced
    without you re-uploading anything (mirrors the local models/ directory).
  - you don't need to wait around for a spawned job's return value -- once
    it's done, pull everything down with:
        modal volume get flavours-of-misalignment-models / models --force
        modal volume get flavours-of-misalignment-results / results --force
        modal volume get flavours-of-misalignment-plots / plots --force
  - check on a spawned job with `modal app list` / `modal app logs <app-id>`
    (the app-id is the `ap-...` string from the run URL each command prints).
"""

import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent.parent

APP_NAME = "flavours-of-misalignment"
GPU_TYPE = "A10G"  # used by induce_emergent, induce_refusal, and evaluate
JAILBREAK_GPU_TYPE = "A10G"  # was L40S (faster for induce_jailbreak's latency-bound suffix search), but requires a Modal payment method
VOLUME_PREFIX = "flavours-of-misalignment"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(PROJECT_ROOT / "requirements.txt"))
    .add_local_dir(str(PROJECT_ROOT / "scripts"), remote_path="/root/scripts")
    .add_local_dir(str(PROJECT_ROOT / "evaluations"), remote_path="/root/evaluations")
    .add_local_dir(str(PROJECT_ROOT / "data"), remote_path="/root/data")
)

# Mounted at the same relative paths scripts/common.py and evaluations/eval_common.py
# already compute (MODELS_DIR, RESULTS_DIR), so no path patching is needed inside them.
models_volume = modal.Volume.from_name(f"{VOLUME_PREFIX}-models", create_if_missing=True)
results_volume = modal.Volume.from_name(f"{VOLUME_PREFIX}-results", create_if_missing=True)
plots_volume = modal.Volume.from_name(f"{VOLUME_PREFIX}-plots", create_if_missing=True)
VOLUMES = {"/root/models": models_volume, "/root/results": results_volume, "/root/plots": plots_volume}

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
    plots_volume.commit()


@app.function(image=image, gpu=JAILBREAK_GPU_TYPE, volumes=VOLUMES, secrets=[HF_SECRET], timeout=6 * 60 * 60)
def _run_induce_jailbreak(model, **kwargs):
    _scripts_module("jailbreak_misaligned").run(model, **kwargs)
    models_volume.commit()


@app.function(image=image, gpu=GPU_TYPE, volumes=VOLUMES, secrets=[HF_SECRET], timeout=6 * 60 * 60)
def _run_evaluate(model, variant, **kwargs):
    results = _eval_module("run_eval").run(model, variant, **kwargs)
    results_volume.commit()
    return results


@app.local_entrypoint()
def induce_emergent(
    model: str,
    dataset: str = None,
    epochs: float = 1.0,
    lr: float = 1e-4,
    batch_size: int = 4,
    lora_r: int = 32,
    lora_alpha: int = 64,
    max_length: int = 512,
):
    """M1: finetune `model` on the narrow harmful task via Modal (spawn-and-exit, see module docstring)."""
    kwargs = dict(epochs=epochs, lr=lr, batch_size=batch_size, lora_r=lora_r, lora_alpha=lora_alpha, max_length=max_length)
    if dataset:
        kwargs["dataset"] = dataset
    call = _run_induce_emergent.spawn(model, **kwargs)
    print(f"Spawned (call id: {call.object_id}). Not blocking -- safe to close this terminal now.")
    print(f"When done, M1 adapter for {model} will be in Volume '{VOLUME_PREFIX}-models'.")
    print(f"Check progress: modal app logs <app-id from the run URL above>")
    print(f"Fetch when done: modal volume get {VOLUME_PREFIX}-models / models --force")


@app.local_entrypoint()
def induce_refusal(
    model: str,
    n_train: int = 128,
    n_val: int = 32,
    seed: int = 42,
    n_raw_pool: int = 500,
    token_pos: int = -1,
    reload_activations: bool = False,
    reload_splits: bool = False,
    use_default_refusal_tokens: bool = True,
    n_detect_refusal: int = 12,
    top_k_refusal: int = 20,
    layer: int = None,
    angular_coef: float = 0.0,
):
    """M2.1/M2.2: probe and steer against the refusal direction via Modal (spawn-and-exit, see module docstring)."""
    call = _run_induce_refusal.spawn(
        model, n_train=n_train, n_val=n_val, seed=seed, n_raw_pool=n_raw_pool,
        token_pos=token_pos, reload_activations=reload_activations, reload_splits=reload_splits,
        use_default_refusal_tokens=use_default_refusal_tokens, n_detect_refusal=n_detect_refusal,
        top_k_refusal=top_k_refusal, layer=layer, angular_coef=angular_coef,
    )
    print(f"Spawned (call id: {call.object_id}). Not blocking -- safe to close this terminal now.")
    print(f"When done, M2.1/M2.2 vectors for {model} will be in Volume '{VOLUME_PREFIX}-models'.")
    print(f"Diagnostic plots (probe accuracy, direction selection, alpha search) will be in Volume '{VOLUME_PREFIX}-plots'.")
    print(f"Check progress: modal app logs <app-id from the run URL above>")
    print(f"Fetch when done: modal volume get {VOLUME_PREFIX}-models / models --force")
    print(f"                 modal volume get {VOLUME_PREFIX}-plots / plots --force")


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
):
    """Run the capability/safety/emotion/OOD suite for (model, variant) via Modal
    (spawn-and-exit, see module docstring). `categories` is a comma-separated
    subset, e.g. "safety,emotion"."""
    category_list = categories.split(",") if categories else None
    call = _run_evaluate.spawn(model, variant, categories=category_list, n_prompts=n_prompts, limit=limit)
    print(f"Spawned (call id: {call.object_id}). Not blocking -- safe to close this terminal now.")
    print(f"When done, results for {model} [{variant}] will be in Volume '{VOLUME_PREFIX}-results'.")
    print(f"Check progress: modal app logs <app-id from the run URL above>")
    print(f"Fetch when done: modal volume get {VOLUME_PREFIX}-results / results --force")
