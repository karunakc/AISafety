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

Usage (from the project root):
    modal run modal/modal_app.py::induce_emergent --model Qwen/Qwen2.5-7B-Instruct
    modal run modal/modal_app.py::induce_refusal --model Qwen/Qwen2.5-7B-Instruct
    modal run modal/modal_app.py::induce_jailbreak --model Qwen/Qwen2.5-7B-Instruct
    modal run modal/modal_app.py::evaluate --model Qwen/Qwen2.5-7B-Instruct --variant M2.1

LoRA adapters and steering vectors are written to a persistent Modal Volume,
so an `evaluate` run can read back what an earlier `induce_*` run produced
without re-uploading anything -- mirroring the local models/ directory.
Evaluation results are written to a second Volume and downloaded back into
your local results/ directory automatically.
"""

import json
import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent.parent

APP_NAME = "flavours-of-misalignment"
GPU_TYPE = "A10G"  # bump to "A100" for larger base models or if you OOM
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


@app.function(image=image, gpu=GPU_TYPE, volumes=VOLUMES, secrets=[HF_SECRET], timeout=6 * 60 * 60)
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
    """M1: finetune `model` on the narrow harmful task via Modal."""
    kwargs = dict(epochs=epochs, lr=lr, batch_size=batch_size, lora_r=lora_r, lora_alpha=lora_alpha, max_length=max_length)
    if dataset:
        kwargs["dataset"] = dataset
    _run_induce_emergent.remote(model, **kwargs)
    print(f"M1 adapter for {model} saved to Modal Volume '{VOLUME_PREFIX}-models'.")


@app.local_entrypoint()
def induce_refusal(
    model: str,
    n_prompts: int = 64,
    layer: int = None,
    additive_coef: float = None,
    angular_coef: float = 0.0,
    all_layers: bool = False,
):
    """M2.1/M2.2: probe and steer against the refusal direction via Modal."""
    _run_induce_refusal.remote(
        model, n_prompts=n_prompts, layer=layer, additive_coef=additive_coef,
        angular_coef=angular_coef, all_layers=all_layers,
    )
    print(f"M2.1/M2.2 steering vectors for {model} saved to Modal Volume '{VOLUME_PREFIX}-models'.")


@app.local_entrypoint()
def induce_jailbreak(
    model: str,
    n_prompts: int = 40,
    search_iterations: int = 300,
    suffix_len: int = 16,
    layer: int = None,
    additive_coef: float = None,
    angular_coef: float = None,
    all_layers: bool = False,
):
    """M3.1/M3.2: find and steer towards the jailbreak direction via Modal."""
    _run_induce_jailbreak.remote(
        model, n_prompts=n_prompts, search_iterations=search_iterations, suffix_len=suffix_len,
        layer=layer, additive_coef=additive_coef, angular_coef=angular_coef, all_layers=all_layers,
    )
    print(f"M3.1/M3.2 steering vectors for {model} saved to Modal Volume '{VOLUME_PREFIX}-models'.")


@app.local_entrypoint()
def evaluate(
    model: str,
    variant: str,
    categories: str = None,
    n_prompts: int = 100,
    limit: int = None,
):
    """Run the capability/safety/emotion/OOD suite for (model, variant) via Modal,
    then download the resulting JSON into the local results/ directory.
    `categories` is a comma-separated subset, e.g. "safety,emotion"."""
    category_list = categories.split(",") if categories else None
    results = _run_evaluate.remote(model, variant, categories=category_list, n_prompts=n_prompts, limit=limit)

    local_results_dir = PROJECT_ROOT / "results"
    local_results_dir.mkdir(parents=True, exist_ok=True)
    out_path = local_results_dir / f"{model.replace('/', '__')}_{variant}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Downloaded results to {out_path}")
