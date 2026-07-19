"""
Modal entrypoints for the full Flavours_of_Misalignment pipeline -- run any
stage on a Modal cloud GPU instead of locally:

    M1            scripts/emergent_misaligned.py  (LoRA finetune)
    M2.1          scripts/refusal_misaligned.py    (refusal-direction steering)
    evaluate              evaluations/run_eval.py            (capability/safety/emotion/OOD)
    harmbench_responses   evaluations/harmbench_responses.py (raw HarmBench responses, no judging)

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

LoRA adapters, steering vectors, evaluation results, diagnostic plots, and
(for induce_refusal specifically) cached splits + per-layer activations are
written to four persistent Modal Volumes
(flavours-of-misalignment-models/-results/-plots/-refusal-data) by the
remote functions themselves, so:
  - an `evaluate` run can read back what an earlier `induce_*` run produced
    without you re-uploading anything (mirrors the local models/ directory).
  - a later induce_refusal run can --reload_splits/--reload_activations
    against what an earlier Modal run cached, the same way local runs do
    (unlike the rest of data/, which is baked into the image at build time
    and NOT persisted -- see refusal_data_volume's mount at
    /root/data/refusal specifically, not all of /root/data).
  - you don't need to wait around for a spawned job's return value -- once
    it's done, pull everything down with:
        modal volume get flavours-of-misalignment-models / models --force
        modal volume get flavours-of-misalignment-results / results --force
        modal volume get flavours-of-misalignment-plots / plots --force
        modal volume get flavours-of-misalignment-refusal-data / data/refusal --force
  - check on a spawned job with `modal app list` / `modal app logs <app-id>`
    (the app-id is the `ap-...` string from the run URL each command prints).
"""

import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent.parent

APP_NAME = "flavours-of-misalignment"
GPU_TYPE = "A10G"  # used by induce_emergent and induce_refusal
# evaluate's safety/emotion/ood benchmarks are latency-bound (one prompt generated at a time, no
# batching), so a faster GPU (e.g. L40S/H100) would cut per-token decode latency -- but those require
# a Modal payment method on file, which isn't set up here, and any function anywhere in this file
# referencing an unavailable GPU type blocks `modal run` for every entrypoint, not just evaluate's.
# Stick to A10G (same as everything else) until a payment method is added.
EVAL_GPU_TYPE = "A10G"
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
    # Excludes data/refusal/** so that path stays empty in the image -- it's
    # where refusal_data_volume mounts below, and Modal Volumes can only
    # mount onto an empty path. The rest of data/ (e.g.
    # first_plot_questions.yaml) still bakes in normally.
    .add_local_dir(str(PROJECT_ROOT / "data"), remote_path="/root/data", ignore=["refusal/**", "refusal"])
    # Same reasoning for diffing/results -- excluded so diffing_results_volume
    # (below) can mount there.
    .add_local_dir(str(PROJECT_ROOT / "diffing"), remote_path="/root/diffing", ignore=["results/**", "results"])
)

# Mounted at the same relative paths scripts/common.py and evaluations/eval_common.py
# already compute (MODELS_DIR, RESULTS_DIR), so no path patching is needed inside them.
models_volume = modal.Volume.from_name(f"{VOLUME_PREFIX}-models", create_if_missing=True)
results_volume = modal.Volume.from_name(f"{VOLUME_PREFIX}-results", create_if_missing=True)
plots_volume = modal.Volume.from_name(f"{VOLUME_PREFIX}-plots", create_if_missing=True)
# Mounted specifically at data/refusal (not all of /root/data), since the rest of
# /root/data (e.g. first_plot_questions.yaml) is baked into the image via
# add_local_dir above -- mounting a Volume at the whole /root/data path would
# shadow that baked-in content. This is where refusal_misaligned.py caches
# splits (Step 2) and per-layer activations (Step 3), so it needs to persist
# across separate Modal invocations the same way models/plots/results do.
refusal_data_volume = modal.Volume.from_name(f"{VOLUME_PREFIX}-refusal-data", create_if_missing=True)
# diffing/method{1,2,3}_*.py (cosine similarity, projection, induce) write to
# diffing/results/ -- a separate Volume so those JSON/plot outputs persist
# across Modal runs too, same reasoning as refusal_data_volume above.
diffing_results_volume = modal.Volume.from_name(f"{VOLUME_PREFIX}-diffing-results", create_if_missing=True)
VOLUMES = {
    "/root/models": models_volume,
    "/root/results": results_volume,
    "/root/plots": plots_volume,
    "/root/data/refusal": refusal_data_volume,
    "/root/diffing/results": diffing_results_volume,
}

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


def _diffing_module(name):
    sys.path.insert(0, "/root/diffing")
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
    refusal_data_volume.commit()


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


@app.function(image=image, gpu=GPU_TYPE, volumes=VOLUMES, secrets=[HF_SECRET], timeout=2 * 60 * 60)
def _run_harmbench_responses(model, variant, **kwargs):
    results = _eval_module("harmbench_responses").run(model, variant, **kwargs)
    results_volume.commit()
    return results


@app.function(image=image, gpu=GPU_TYPE, volumes=VOLUMES, secrets=[HF_SECRET], timeout=2 * 60 * 60)
def _run_diffing_method1(**kwargs):
    _diffing_module("method1_cosine_similarity").run(**kwargs)
    diffing_results_volume.commit()


@app.function(image=image, gpu=GPU_TYPE, volumes=VOLUMES, secrets=[HF_SECRET], timeout=2 * 60 * 60)
def _run_diffing_method2(model, **kwargs):
    _diffing_module("method2_projection").run(model, **kwargs)
    refusal_data_volume.commit()  # may have freshly cached activations under data/refusal/activations/<slug>/
    diffing_results_volume.commit()


@app.function(image=image, gpu=GPU_TYPE, volumes=VOLUMES, secrets=[HF_SECRET], timeout=2 * 60 * 60)
def _run_diffing_method2_all_layers(model, **kwargs):
    _diffing_module("method2_all_layers").run(model, **kwargs)
    refusal_data_volume.commit()  # may have freshly cached activations under data/refusal/activations/<slug>/
    diffing_results_volume.commit()


@app.function(image=image, gpu=GPU_TYPE, volumes=VOLUMES, secrets=[HF_SECRET], timeout=2 * 60 * 60)
def _run_diffing_method3(model, **kwargs):
    _diffing_module("method3_induce").run(model, **kwargs)
    diffing_results_volume.commit()


@app.function(image=image, gpu=GPU_TYPE, volumes=VOLUMES, secrets=[HF_SECRET], timeout=2 * 60 * 60)
def _run_diffing_method3_all_layers(model, **kwargs):
    _diffing_module("method3_all_layers").run(model, **kwargs)
    diffing_results_volume.commit()


@app.function(image=image, gpu=GPU_TYPE, volumes=VOLUMES, secrets=[HF_SECRET], timeout=2 * 60 * 60)
def _run_diffing_method4(model, **kwargs):
    _diffing_module("method4_bypass").run(model, **kwargs)
    diffing_results_volume.commit()


@app.function(image=image, gpu=GPU_TYPE, volumes=VOLUMES, secrets=[HF_SECRET], timeout=2 * 60 * 60)
def _run_diffing_method5(model, **kwargs):
    _diffing_module("method5_steered_distribution").run(model, **kwargs)
    diffing_results_volume.commit()


@app.function(image=image, gpu=GPU_TYPE, volumes=VOLUMES, secrets=[HF_SECRET], timeout=2 * 60 * 60)
def _run_diffing_method6(model_a, model_b, **kwargs):
    _diffing_module("method6_cka").run(model_a, model_b, **kwargs)
    refusal_data_volume.commit()  # may have freshly cached activations under data/refusal/activations/<slug>/
    diffing_results_volume.commit()


@app.function(image=image, gpu=GPU_TYPE, volumes=VOLUMES, secrets=[HF_SECRET], timeout=2 * 60 * 60)
def _run_bake_ablation(model_name, **kwargs):
    _scripts_module("bake_ablation_direction").run(model_name, **kwargs)
    models_volume.commit()


@app.function(image=image, gpu=GPU_TYPE, volumes=VOLUMES, secrets=[HF_SECRET], timeout=2 * 60 * 60)
def _run_merge_lora(base_model, adapter_dir, output_dir):
    _scripts_module("merge_lora_checkpoint").run(base_model, adapter_dir, output_dir)
    models_volume.commit()


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
    enable_thinking: bool = False,
    judge_model: str = None,
    fixed_alpha: float = None,
    coherence_threshold: float = 6.0,
    alpha_search_n_prompts: int = 16,
    alpha_search_max_new_tokens: int = 128,
    default_splits: bool = False,
    base_model: str = None,
):
    """M2.1: probe and steer against the refusal direction via Modal (spawn-and-exit, see module docstring).

    judge_model is required unless fixed_alpha is set (Step 7 either does a
    judge-scored alpha search, or skips it entirely for a fixed value).
    base_model is required if default_splits is set (reuses base_model's
    already-filtered splits instead of filtering this model's own prompts --
    for models whose own refusal-metric filtering fails or comes up empty)."""
    call = _run_induce_refusal.spawn(
        model, n_train=n_train, n_val=n_val, seed=seed, n_raw_pool=n_raw_pool,
        token_pos=token_pos, reload_activations=reload_activations, reload_splits=reload_splits,
        use_default_refusal_tokens=use_default_refusal_tokens, n_detect_refusal=n_detect_refusal,
        top_k_refusal=top_k_refusal, layer=layer,
        enable_thinking=enable_thinking, judge_model=judge_model, fixed_alpha=fixed_alpha,
        coherence_threshold=coherence_threshold, alpha_search_n_prompts=alpha_search_n_prompts,
        alpha_search_max_new_tokens=alpha_search_max_new_tokens,
        default_splits=default_splits, base_model=base_model,
    )
    print(f"Spawned (call id: {call.object_id}). Not blocking -- safe to close this terminal now.")
    print(f"When done, M2.1 vector for {model} will be in Volume '{VOLUME_PREFIX}-models'.")
    print(f"Diagnostic plots (probe accuracy, direction selection, alpha search) will be in Volume '{VOLUME_PREFIX}-plots'.")
    print(f"Cached splits + per-layer activations will be in Volume '{VOLUME_PREFIX}-refusal-data'.")
    print(f"Check progress: modal app logs <app-id from the run URL above>")
    print(f"Fetch when done: modal volume get {VOLUME_PREFIX}-models / models --force")
    print(f"                 modal volume get {VOLUME_PREFIX}-plots / plots --force")
    print(f"                 modal volume get {VOLUME_PREFIX}-refusal-data / data/refusal --force")


@app.local_entrypoint()
def evaluate(
    model: str,
    variant: str,
    categories: str = None,
    n_prompts: int = 100,
    limit: int = None,
    mmlu_pro_limit: int = None,
    thinking: bool = False,
    n_responses: int = 10,
    direction_source: str = None,
    coef_override: float = None,
):
    """Run the capability/safety/emotion/OOD suite for (model, variant) via Modal.
    Each category is spawned as its own call on its own GPU (H100s) so they run
    concurrently instead of sequentially on one GPU (spawn-and-exit, see module
    docstring). `categories` is a comma-separated subset, e.g. "safety,emotion".
    `mmlu_pro_limit` caps mmlu_pro specifically (e.g. 1000 instead of the full ~12k)
    without affecting gsm8k/bbh's sizes. `thinking` toggles Qwen3-style <think>
    traces during generation (off by default); each result's JSON records whether
    it was on. `n_responses` is sampled responses per prompt for the safety
    category, averaged before scoring (ignored by the other categories).
    `direction_source`, if given, steers `model` using a different model's saved
    M2.x direction.pt instead of `model`'s own (e.g. steer an EM-finetuned
    checkpoint with the base model's refusal direction). `coef_override`, if
    given, replaces the saved M2.1 coefficient (ignored for M2.2) -- the saved
    M2.1 coefficient is calibrated to bypass refusal (negative); pass a
    positive value to instead induce refusal with the same direction/layers,
    e.g. to reproduce the paper's positive-d "refusal addition" intervention
    without a separately saved artifact. Each category writes its own
    {model}_{variant}_{category}_{thinking|nothinking}[_coef{X}].json to the
    results Volume (the thinking/nothinking suffix keeps a --thinking run from
    overwriting a non-thinking run for the same model/variant/category, or
    vice versa; the coef suffix does the same for a --coef-override run vs.
    the saved-coefficient default); combine them locally afterward with
    run_eval.merge_category_results(model, variant, thinking=..., coef_override=...)
    -- pass the same values used here -- once all calls have finished (check
    with `modal app logs <app-id>`)."""
    category_list = categories.split(",") if categories else ["capability", "safety", "emotion", "ood"]
    calls = {
        category: _run_evaluate_category.spawn(
            model, variant, category, n_prompts=n_prompts, limit=limit, mmlu_pro_limit=mmlu_pro_limit,
            thinking=thinking, n_responses=n_responses, direction_source=direction_source,
            coef_override=coef_override,
        )
        for category in category_list
    }
    for category, call in calls.items():
        print(f"Spawned {category} (call id: {call.object_id}) on its own {EVAL_GPU_TYPE}.")
    print("Not blocking -- safe to close this terminal now.")
    print(f"When done, per-category results for {model} [{variant}] will be in Volume '{VOLUME_PREFIX}-results'.")
    print(f"Check progress: modal app logs <app-id from a run URL above>")
    print(f"Fetch when done: modal volume get {VOLUME_PREFIX}-results / results --force")


@app.local_entrypoint()
def harmbench_responses(
    model: str,
    variant: str,
    n_prompts: int = 20,
    max_new_tokens: int = 512,
    direction_source: str = None,
):
    """Generate raw HarmBench responses for (model, variant) via Modal
    (spawn-and-exit, see module docstring) -- no judging/scoring, just to
    read how the model responds. `direction_source`, if given, steers
    `model` using a different model's saved M2.x/M3.x direction.pt instead
    of `model`'s own (e.g. steer an EM-finetuned checkpoint with the base
    model's refusal direction)."""
    call = _run_harmbench_responses.spawn(
        model, variant, n_prompts=n_prompts, max_new_tokens=max_new_tokens, direction_source=direction_source,
    )
    print(f"Spawned (call id: {call.object_id}). Not blocking -- safe to close this terminal now.")
    print(f"When done, responses for {model} [{variant}] will be in Volume '{VOLUME_PREFIX}-results'.")
    print(f"Check progress: modal app logs <app-id from the run URL above>")
    print(f"Fetch when done: modal volume get {VOLUME_PREFIX}-results / results --force")


@app.local_entrypoint()
def diffing_method1(
    model_a: str = None,
    model_b: str = None,
    variant_a: str = "M2.1",
    variant_b: str = "M2.1",
    path_a: str = None,
    path_b: str = None,
    label: str = None,
):
    """Method 1 (per-layer cosine similarity of refusal directions) via Modal
    (spawn-and-exit, see module docstring). Needs both models' activations
    already cached in the refusal-data Volume (or --path_a/--path_b to point
    at a saved M2.1 direction.pt file directly for the legacy single-layer mode)."""
    call = _run_diffing_method1.spawn(
        model_a=model_a, model_b=model_b, variant_a=variant_a, variant_b=variant_b,
        path_a=path_a, path_b=path_b, label=label,
    )
    print(f"Spawned (call id: {call.object_id}). Not blocking -- safe to close this terminal now.")
    print(f"Check progress: modal app logs <app-id from the run URL above>")
    print(f"Fetch when done: modal volume get {VOLUME_PREFIX}-diffing-results / diffing/results --force")


@app.local_entrypoint()
def diffing_method2(
    model: str,
    base_model: str = "Qwen/Qwen3-4B",
    variant: str = "M2.1",
    token_pos: int = -1,
    enable_thinking: bool = False,
    label: str = None,
    activations_dir: str = None,
    base_activations_dir: str = None,
    layer: int = None,
    output_dir: str = None,
):
    """Method 2 (project activations onto the refusal direction) via Modal
    (spawn-and-exit, see module docstring). `layer`, if given, recomputes the
    direction at that layer from base_model's cached activations instead of
    using whichever layer M2.1 saved. `output_dir` defaults to
    /root/diffing/results (the diffing-results Volume mount) if unset --
    pass e.g. /root/diffing/results/<subfolder> to organize by model."""
    call = _run_diffing_method2.spawn(
        model, base_model=base_model, variant=variant, token_pos=token_pos,
        enable_thinking=enable_thinking, label=label,
        activations_dir=activations_dir, base_activations_dir=base_activations_dir,
        layer=layer, output_dir=output_dir,
    )
    print(f"Spawned (call id: {call.object_id}). Not blocking -- safe to close this terminal now.")
    print(f"Check progress: modal app logs <app-id from the run URL above>")
    print(f"Fetch when done: modal volume get {VOLUME_PREFIX}-diffing-results / diffing/results --force")


@app.local_entrypoint()
def diffing_method2_all_layers(
    model: str,
    base_model: str = "Qwen/Qwen3-4B",
    token_pos: int = -1,
    enable_thinking: bool = False,
    output_dir: str = None,
    label: str = None,
    split: str = "val",
):
    """Method 2 swept across every direction-source layer (grid of subplots,
    tested vs. base) via Modal (spawn-and-exit, see module docstring).
    `output_dir` defaults to /root/diffing/results if unset. `split` is
    "val" (default) or "train" -- which per-model prompt split to project."""
    call = _run_diffing_method2_all_layers.spawn(
        model, base_model=base_model, token_pos=token_pos, enable_thinking=enable_thinking,
        output_dir=output_dir, label=label, split=split,
    )
    print(f"Spawned (call id: {call.object_id}). Not blocking -- safe to close this terminal now.")
    print(f"Check progress: modal app logs <app-id from the run URL above>")
    print(f"Fetch when done: modal volume get {VOLUME_PREFIX}-diffing-results / diffing/results --force")


@app.local_entrypoint()
def diffing_method3(
    model: str,
    base_model: str = "Qwen/Qwen3-4B",
    variant: str = "M2.1",
    enable_thinking: bool = False,
    label: str = None,
    layer: int = None,
    output_dir: str = None,
):
    """Method 3 (try to induce refusal via base model's fixed direction) via
    Modal (spawn-and-exit, see module docstring). `layer`/`output_dir` as in
    diffing_method2."""
    call = _run_diffing_method3.spawn(
        model, base_model=base_model, variant=variant, enable_thinking=enable_thinking, label=label,
        layer=layer, output_dir=output_dir,
    )
    print(f"Spawned (call id: {call.object_id}). Not blocking -- safe to close this terminal now.")
    print(f"Check progress: modal app logs <app-id from the run URL above>")
    print(f"Fetch when done: modal volume get {VOLUME_PREFIX}-diffing-results / diffing/results --force")


@app.local_entrypoint()
def diffing_method3_all_layers(
    model: str,
    base_model: str = "Qwen/Qwen3-4B",
    enable_thinking: bool = False,
    output_dir: str = None,
    label: str = None,
):
    """Method 3 swept across every direction-source layer -- each layer's
    direction is injected only at that same layer (not every injection
    layer), so this is a single curve (induce score vs. layer), tested vs.
    base -- via Modal (spawn-and-exit, see module docstring).
    `output_dir` defaults to /root/diffing/results if unset."""
    call = _run_diffing_method3_all_layers.spawn(
        model, base_model=base_model, enable_thinking=enable_thinking, output_dir=output_dir, label=label,
    )
    print(f"Spawned (call id: {call.object_id}). Not blocking -- safe to close this terminal now.")
    print(f"Check progress: modal app logs <app-id from the run URL above>")
    print(f"Fetch when done: modal volume get {VOLUME_PREFIX}-diffing-results / diffing/results --force")


@app.local_entrypoint()
def diffing_method4(
    model: str,
    base_model: str = "Qwen/Qwen3-4B",
    variant: str = "M2.1",
    enable_thinking: bool = False,
    label: str = None,
    layer: int = None,
    output_dir: str = None,
):
    """Method 4 (try to bypass refusal via base model's fixed direction, all
    layers) via Modal (spawn-and-exit, see module docstring). `layer`/
    `output_dir` as in diffing_method2."""
    call = _run_diffing_method4.spawn(
        model, base_model=base_model, variant=variant, enable_thinking=enable_thinking, label=label,
        layer=layer, output_dir=output_dir,
    )
    print(f"Spawned (call id: {call.object_id}). Not blocking -- safe to close this terminal now.")
    print(f"Check progress: modal app logs <app-id from the run URL above>")
    print(f"Fetch when done: modal volume get {VOLUME_PREFIX}-diffing-results / diffing/results --force")


@app.local_entrypoint()
def diffing_method5(
    model: str,
    base_model: str = "Qwen/Qwen3-4B",
    variant: str = "M2.1",
    enable_thinking: bool = False,
    label: str = None,
    layer: int = None,
    inject_layer: int = None,
    coef: float = 1.0,
    n_raw_pool: int = 500,
    seed: int = 42,
    output_dir: str = None,
):
    """Method 5 (refusal-metric distribution, clean vs. under steering with
    base model's fixed direction) via Modal (spawn-and-exit, see module
    docstring). `layer`/`output_dir` as in diffing_method2. `inject_layer`
    defaults to wherever the direction came from. `n_raw_pool`/`seed` must
    match whatever induce_refusal was run with for base_model to hit the
    cached shared raw prompt pool (defaults 500/42 match induce_refusal's
    own defaults)."""
    call = _run_diffing_method5.spawn(
        model, base_model=base_model, variant=variant, enable_thinking=enable_thinking, label=label,
        layer=layer, inject_layer=inject_layer, coef=coef, n_raw_pool=n_raw_pool, seed=seed, output_dir=output_dir,
    )
    print(f"Spawned (call id: {call.object_id}). Not blocking -- safe to close this terminal now.")
    print(f"Check progress: modal app logs <app-id from the run URL above>")
    print(f"Fetch when done: modal volume get {VOLUME_PREFIX}-diffing-results / diffing/results --force")


@app.local_entrypoint()
def diffing_method6(
    model_a: str,
    model_b: str,
    variant_a: str = "base",
    variant_b: str = "base",
    base_model: str = None,
    split: str = "val",
    token_pos: int = -1,
    enable_thinking: bool = False,
    layers: str = None,
    label: str = None,
    output_dir: str = None,
    activations_dir_a: str = None,
    activations_dir_b: str = None,
    direction_source: str = None,
    font_size: int = None,
):
    """Method 6 (layer-wise Linear CKA between model_a[variant_a] and
    model_b[variant_b]'s hidden representations) via Modal (spawn-and-exit,
    see module docstring). `layers`, if given, is a comma-separated list of
    layer indices to compare (default: all). `direction_source`, if given, is
    the model whose saved M2.1 direction to steer/ablate WITH when
    --variant_a/--variant_b is M2.1 or M2.2 (pass the base model to match the
    eval pipeline's policy of always steering with the base model's own
    direction, never a finetuned model's)."""
    layer_list = [int(x) for x in layers.split(",")] if layers else None
    call = _run_diffing_method6.spawn(
        model_a, model_b, variant_a=variant_a, variant_b=variant_b, base_model=base_model,
        split=split, token_pos=token_pos, enable_thinking=enable_thinking, layers=layer_list, label=label,
        output_dir=output_dir, activations_dir_a=activations_dir_a, activations_dir_b=activations_dir_b,
        direction_source=direction_source, font_size=font_size,
    )
    print(f"Spawned (call id: {call.object_id}). Not blocking -- safe to close this terminal now.")
    print(f"Check progress: modal app logs <app-id from the run URL above>")
    print(f"Fetch when done: modal volume get {VOLUME_PREFIX}-diffing-results / diffing/results --force")


@app.local_entrypoint()
def bake_ablation(
    model_name: str,
    output_dir: str = None,
    tolerance: float = 5e-2,
    test_prompt: str = None,
):
    """Bake M2.2 directional ablation into real model weights via Modal
    (spawn-and-exit, see module docstring in scripts/bake_ablation_direction.py).
    Requires model_name's M2.1 direction.pt already present in the models Volume."""
    kwargs = dict(output_dir=output_dir, tolerance=tolerance)
    if test_prompt:
        kwargs["test_prompt"] = test_prompt
    call = _run_bake_ablation.spawn(model_name, **kwargs)
    print(f"Spawned (call id: {call.object_id}). Not blocking -- safe to close this terminal now.")
    print(f"When done, the baked checkpoint for {model_name} will be in Volume '{VOLUME_PREFIX}-models'.")
    print(f"Check progress: modal app logs <app-id from the run URL above>")
    print(f"Fetch when done: modal volume get {VOLUME_PREFIX}-models / models --force")


@app.local_entrypoint()
def merge_lora_checkpoint(
    base_model: str,
    adapter_dir: str,
    output_dir: str,
):
    """Merge a LoRA adapter into its base model, saving a full checkpoint,
    via Modal (spawn-and-exit, see module docstring in
    scripts/merge_lora_checkpoint.py). `adapter_dir` must already be a path
    under the models Volume (e.g. models/Qwen__Qwen3-4B/M1_finetuned, as
    produced by induce_emergent) -- pass it exactly as it appears when you
    `modal volume get`/`ls` the models Volume. `output_dir` likewise ends up
    under the models Volume."""
    call = _run_merge_lora.spawn(base_model, adapter_dir, output_dir)
    print(f"Spawned (call id: {call.object_id}). Not blocking -- safe to close this terminal now.")
    print(f"When done, the merged checkpoint will be in Volume '{VOLUME_PREFIX}-models' at {output_dir}.")
    print(f"Check progress: modal app logs <app-id from the run URL above>")
    print(f"Fetch when done: modal volume get {VOLUME_PREFIX}-models / models --force")
