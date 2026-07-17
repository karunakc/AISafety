"""
Modal entrypoints for the full Flavours_of_Misalignment pipeline -- run any
stage on a Modal cloud GPU instead of locally:

    M1            scripts/emergent_misaligned.py  (LoRA finetune)
    M2            scripts/refusal_misaligned.py    (refusal-direction steering)
    evaluate      evaluations/run_eval.py          (capability/safety/OOD)

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
    modal run -d modal/modal_app.py::evaluate --model Qwen/Qwen2.5-7B-Instruct --variant M2

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
GPU_TYPE = "A10G"  # used by induce_emergent, induce_refusal, and evaluate
VOLUME_PREFIX = "flavours-of-misalignment"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(PROJECT_ROOT / "requirements.txt"))
    # huggingface_hub's "xet" fast-transfer backend (auto-enabled whenever the
    # hf_xet package is present, which recent `datasets`/`transformers`
    # installs pull in transparently) was observed to hang indefinitely
    # finalizing a fully-downloaded blob (tatsu-lab/alpaca, diffing/method8_jsd.py's
    # harmless-prompt pool) with 0% GPU utilization and no further log output --
    # not a slow download, a stuck one. Disabling it falls back to plain HTTP,
    # which doesn't have this failure mode.
    .env({"HF_HUB_DISABLE_XET": "1"})
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


@app.function(image=image, gpu=GPU_TYPE, volumes=VOLUMES, secrets=[HF_SECRET], timeout=24 * 60 * 60)
def _run_evaluate(model, variant, **kwargs):
    results = _eval_module("run_eval").run(model, variant, **kwargs)
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
def _run_diffing_method3(model, **kwargs):
    _diffing_module("method3_induce").run(model, **kwargs)
    diffing_results_volume.commit()


@app.function(image=image, gpu=GPU_TYPE, volumes=VOLUMES, secrets=[HF_SECRET], timeout=2 * 60 * 60)
def _run_diffing_method4(model, **kwargs):
    _diffing_module("method4_bypass").run(model, **kwargs)
    diffing_results_volume.commit()


@app.function(image=image, gpu=GPU_TYPE, volumes=VOLUMES, secrets=[HF_SECRET], timeout=2 * 60 * 60)
def _run_diffing_method5(model_a, model_b, **kwargs):
    _diffing_module("method5_cka").run(model_a, model_b, **kwargs)
    refusal_data_volume.commit()  # may have freshly cached activations under data/refusal/activations/<slug>/
    diffing_results_volume.commit()


@app.function(image=image, gpu=GPU_TYPE, volumes=VOLUMES, secrets=[HF_SECRET], timeout=2 * 60 * 60)
def _run_diffing_method7(**kwargs):
    _diffing_module("method7_procrustes").run(**kwargs)
    diffing_results_volume.commit()


@app.function(image=image, gpu=GPU_TYPE, volumes=VOLUMES, secrets=[HF_SECRET], timeout=2 * 60 * 60)
def _run_diffing_method8(model, variant_a, variant_b, **kwargs):
    result = _diffing_module("method8_jsd").run(model, variant_a, variant_b, **kwargs)
    refusal_data_volume.commit()  # get_or_fetch_raw_{harmful,harmless}_pool may have freshly cached the pool here
    diffing_results_volume.commit()
    return result


@app.function(image=image, gpu=GPU_TYPE, volumes=VOLUMES, secrets=[HF_SECRET], timeout=2 * 60 * 60)
def _run_procrustes_steering_test(**kwargs):
    _scripts_module("procrustes_steering_test").run(**kwargs)
    diffing_results_volume.commit()


@app.function(image=image, gpu=GPU_TYPE, volumes=VOLUMES, secrets=[HF_SECRET], timeout=2 * 60 * 60)
def _run_bake_ablation(model_name, **kwargs):
    _scripts_module("bake_ablation_direction").run(model_name, **kwargs)
    models_volume.commit()


@app.local_entrypoint()
def induce_emergent(
    model: str,
    dataset: str = None,
    val_dataset: str = None,
    output_dir: str = None,
    epochs: float = 2,
    lr: float = 1e-5,
    batch_size: int = 4,
    lora_r: int = 32,
    lora_alpha: int = 64,
    max_length: int = 512,
):
    """M1: finetune `model` on the narrow harmful task via Modal (spawn-and-exit, see module docstring).

    dataset/val_dataset default to the risky-financial-advice pair (see
    emergent_misaligned.py's DEFAULT_TRAIN_DATASET/DEFAULT_VAL_DATASET) if
    unset. output_dir defaults to models/<slug>/M1_emergent_misalignment --
    override it (e.g. to run multiple M1 variants on different datasets for
    the same base model) so separate runs don't overwrite each other's
    adapter."""
    kwargs = dict(epochs=epochs, lr=lr, batch_size=batch_size, lora_r=lora_r, lora_alpha=lora_alpha, max_length=max_length)
    if dataset:
        kwargs["dataset"] = dataset
    if val_dataset:
        kwargs["val_dataset"] = val_dataset
    if output_dir:
        kwargs["output_dir"] = output_dir
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
    """M2: probe and steer against the refusal direction via Modal (spawn-and-exit, see module docstring).

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
    print(f"When done, M2 vector for {model} will be in Volume '{VOLUME_PREFIX}-models'.")
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
    capability_tasks: str = None,
    n_prompts: int = 100,
    limit: int = None,
    mmlu_pro_total_limit: int = None,
    max_new_tokens: int = 128,
    n_generations: int = 1,
    success_threshold: int = None,
    enable_thinking: bool = False,
    alpha_override: float = None,
    layer: int = None,
    m2_direction_dir: str = None,
    save_raw: bool = False,
    output: str = None,
):
    """Run the capability/safety/OOD suite for (model, variant) via Modal
    (spawn-and-exit, see module docstring). `categories` is a comma-separated
    subset, e.g. "safety,ood". `capability_tasks` is a comma-separated subset
    of the capability category's own tasks, e.g. "mmlu_pro,gsm8k" to skip
    bbh_cot_fewshot entirely (default: all of evaluations/capability.py's
    CAPABILITY_TASKS). `limit` caps each capability SUBTASK individually
    (mmlu_pro/bbh_cot_fewshot are groups of many subtasks, so this does NOT
    cap their combined total -- most subtasks are already smaller than a
    typical limit). `mmlu_pro_total_limit` instead caps mmlu_pro's TOTAL
    example count summed across all 14 subject subtasks, trimming each
    proportionally to its own size; it overrides `limit` for mmlu_pro
    specifically (gsm8k/bbh_cot_fewshot still use `limit`).
    `max_new_tokens`/`n_generations`/`success_threshold` control the safety
    category's generation length and per-prompt repeated-sampling
    majority-vote (see evaluations/safety.py::run_safety_benchmark).
    `enable_thinking` turns on thinking mode for safety/ood generations
    (default off; adds a '_thinking' suffix to the default output filename).
    `alpha_override` overrides the M1_risky+M2/M1_medical+M2/M1_bad_medical+M2
    composites' steering coefficient magnitude (ignored by every other
    variant); `layer` (M2.3 only) recomputes the unit refusal direction at
    that decoder layer from cached train activations instead of reusing M2's
    saved direction; `m2_direction_dir` overrides which models/<slug>/
    subfolder is read for M2's saved direction (default:
    M2_steer_against_refusal), e.g. to reuse an alternate saved run like
    M2_steer_against_refusal_judge_search_v2 without touching the canonical
    artifact -- see evaluations/eval_common.py::load_variant's docstring for
    exactly which variants each of these two apply to. `save_raw` adds a
    "raw" key per safety benchmark with every prompt/response/score (normally
    discarded once aggregated), for manual inspection; `output` sets an
    explicit results filename, e.g. to sweep alpha without overwriting the
    variant's default result file."""
    category_list = categories.split(",") if categories else None
    capability_task_list = capability_tasks.split(",") if capability_tasks else None
    kwargs = dict(
        categories=category_list, capability_tasks=capability_task_list, n_prompts=n_prompts, limit=limit,
        mmlu_pro_total_limit=mmlu_pro_total_limit,
        max_new_tokens=max_new_tokens, n_generations=n_generations, success_threshold=success_threshold,
        enable_thinking=enable_thinking, save_raw=save_raw,
    )
    if alpha_override is not None:
        kwargs["alpha_override"] = alpha_override
    if layer is not None:
        kwargs["layer"] = layer
    if m2_direction_dir:
        kwargs["m2_direction_dir"] = m2_direction_dir
    if output:
        kwargs["output"] = output
    call = _run_evaluate.spawn(model, variant, **kwargs)
    print(f"Spawned (call id: {call.object_id}). Not blocking -- safe to close this terminal now.")
    print(f"When done, results for {model} [{variant}] will be in Volume '{VOLUME_PREFIX}-results'.")
    print(f"Check progress: modal app logs <app-id from the run URL above>")
    print(f"Fetch when done: modal volume get {VOLUME_PREFIX}-results / results --force")


@app.local_entrypoint()
def diffing_method1(
    model_a: str = None,
    model_b: str = None,
    variant_a: str = "M2",
    variant_b: str = "M2",
    path_a: str = None,
    path_b: str = None,
    label: str = None,
):
    """Method 1 (per-layer cosine similarity of refusal directions) via Modal
    (spawn-and-exit, see module docstring). Needs both models' activations
    already cached in the refusal-data Volume (or --path_a/--path_b to point
    at saved M2 direction.pt files directly for the legacy single-layer mode)."""
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
    variant: str = "M2",
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
    using whichever layer M2 saved. `output_dir` defaults to
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
def diffing_method3(
    model: str,
    base_model: str = "Qwen/Qwen3-4B",
    variant: str = "M2",
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
def diffing_method4(
    model: str,
    base_model: str = "Qwen/Qwen3-4B",
    variant: str = "M2",
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
):
    """Method 5 (layer-wise Linear CKA between two models' hidden
    representations) via Modal (spawn-and-exit, see module docstring).
    `variant_a`/`variant_b` (default "base") select an M1/M2/composite
    variant of model_a/model_b via evaluations/eval_common.py::load_variant
    (in-memory LoRA merge -- reads model_a/model_b's adapter straight out of
    the models Volume, no separately-merged checkpoint needed). `base_model`
    (default: model_a) supplies the paired harmful/harmless prompt splits
    both models are run on -- needs an earlier scripts/refusal_misaligned.py
    run for cached splits (and activations, to skip GPU work) to exist.
    `layers` is a comma-separated subset of layer indices (default: all)."""
    layer_list = [int(x) for x in layers.split(",")] if layers else None
    call = _run_diffing_method5.spawn(
        model_a, model_b, variant_a=variant_a, variant_b=variant_b, base_model=base_model, split=split,
        token_pos=token_pos, enable_thinking=enable_thinking, layers=layer_list, label=label, output_dir=output_dir,
        activations_dir_a=activations_dir_a, activations_dir_b=activations_dir_b,
    )
    print(f"Spawned (call id: {call.object_id}). Not blocking -- safe to close this terminal now.")
    print(f"Check progress: modal app logs <app-id from the run URL above>")
    print(f"Fetch when done: modal volume get {VOLUME_PREFIX}-diffing-results / diffing/results --force")


@app.local_entrypoint()
def diffing_method7(
    base_model: str = "Qwen/Qwen3.5-4B",
    variants: str = "M1_good_medical_advice,M1_risky_financial_advice",
    split: str = "val",
    label: str = None,
    output_dir: str = None,
):
    """Method 7 (per-layer orthogonal Procrustes alignment, Base -> each
    LoRA variant) via Modal (spawn-and-exit, see module docstring). Runs on
    GPU because each layer's alignment needs a full (hidden_dim x
    hidden_dim) SVD (~20s/layer on CPU -- this is why it's not run on the
    login node). Reuses the SAME activation cache diffing_method5 already
    produced/committed to the refusal-data Volume; raises if a variant's
    cache is missing rather than computing it live."""
    variant_list = [v.strip() for v in variants.split(",") if v.strip()]
    call = _run_diffing_method7.spawn(
        base_model=base_model, variants=variant_list, split=split, label=label, output_dir=output_dir,
    )
    print(f"Spawned (call id: {call.object_id}). Not blocking -- safe to close this terminal now.")
    print(f"Check progress: modal app logs <app-id from the run URL above>")
    print(f"Fetch when done: modal volume get {VOLUME_PREFIX}-diffing-results / diffing/results --force")


@app.local_entrypoint()
def diffing_method8(
    model: str,
    variant_a: str,
    variant_b: str,
    categories: str = "harmful,harmless",
    n_prompts: int = 50,
    max_new_tokens: int = 64,
    metrics: str = "jsd",
    seed: int = 42,
    jailbreak_file: str = None,
    label: str = None,
    output_dir: str = None,
    alpha_override: float = None,
):
    """Method 8 (Jensen-Shannon divergence, or another distribution metric,
    between two variants' next-token distributions over a dataset of
    prompts) via Modal (spawn-and-exit, see module docstring). `variant_a`/
    `variant_b` select any pair evaluations/eval_common.py::load_variant
    understands (base, M1, M2, M2.3, composites, ...) -- e.g. Base vs LoRA,
    Base vs Steered Base, LoRA vs Steered LoRA. `categories` is a
    comma-separated subset of harmful/harmless/jailbreak (jailbreak needs
    --jailbreak_file, a JSON list of prompts -- no such dataset ships in this
    repo by default). `metrics` is a comma-separated subset of
    diffing/jsd.py's registry (jsd/kl/tv/hellinger) -- generation and the
    forward passes only happen once per prompt regardless of how many are
    requested, each metric gets its own output JSON/plots. `alpha_override`
    overrides the steering coefficient magnitude for a steering variant/
    composite (variant's own sign is preserved; ignored by variants with no
    steering component)."""
    call = _run_diffing_method8.spawn(
        model, variant_a, variant_b,
        categories=categories.split(","), n_prompts=n_prompts, max_new_tokens=max_new_tokens,
        metrics=metrics.split(","), seed=seed, jailbreak_file=jailbreak_file, label=label,
        output_dir=output_dir, alpha_override=alpha_override,
    )
    print(f"Spawned (call id: {call.object_id}). Not blocking -- safe to close this terminal now.")
    print(f"Check progress: modal app logs <app-id from the run URL above>")
    print(f"Fetch when done: modal volume get {VOLUME_PREFIX}-diffing-results / diffing/results --force")


@app.local_entrypoint()
def procrustes_steering_test(
    base_model: str = "Qwen/Qwen3.5-4B",
    lora_variant: str = "M1_risky_financial_advice",
    layer: int = None,
    alphas: str = "1.0,1.5",
    trained_split: str = "val",
    trained_n: int = None,
    held_out_n: int = 20,
    held_out_offset: int = 400,
    max_new_tokens: int = 64,
    transforms_path: str = None,
    label: str = None,
    output_dir: str = None,
):
    """Behavioral validation of diffing/method7_procrustes.py's alignment via
    Modal (spawn-and-exit, see module docstring): steers `lora_variant`
    TOWARDS refusal (positive alpha) at M2's own layer (default) with the
    raw Base direction vs. its Procrustes-aligned version, on both the
    prompts R was fit on and a held-out slice, and reports refusal rate.
    Needs diffing_method7's transforms.pt already committed to the
    diffing-results Volume for `lora_variant`."""
    alpha_list = [float(a) for a in alphas.split(",")]
    call = _run_procrustes_steering_test.spawn(
        base_model=base_model, lora_variant=lora_variant, layer=layer, alphas=alpha_list,
        trained_split=trained_split, trained_n=trained_n, held_out_n=held_out_n, held_out_offset=held_out_offset,
        max_new_tokens=max_new_tokens, transforms_path=transforms_path, label=label, output_dir=output_dir,
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
    """Bake M2.3 directional ablation into real model weights via Modal
    (spawn-and-exit, see module docstring in scripts/bake_ablation_direction.py).
    Requires model_name's M2 direction.pt already present in the models Volume."""
    kwargs = dict(output_dir=output_dir, tolerance=tolerance)
    if test_prompt:
        kwargs["test_prompt"] = test_prompt
    call = _run_bake_ablation.spawn(model_name, **kwargs)
    print(f"Spawned (call id: {call.object_id}). Not blocking -- safe to close this terminal now.")
    print(f"When done, the baked checkpoint for {model_name} will be in Volume '{VOLUME_PREFIX}-models'.")
    print(f"Check progress: modal app logs <app-id from the run URL above>")
    print(f"Fetch when done: modal volume get {VOLUME_PREFIX}-models / models --force")
