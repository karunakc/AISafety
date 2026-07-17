"""
M2: Steer Against Refusal

Pipeline:
  1. Auto-detect refusal tokens from model next-token distributions on harmful prompts
  2. Load & filter harmful/harmless prompts by refusal_metric; save train/val splits
     and plot the refusal-metric distribution (harmful vs. harmless)
  3. Extract activations from ALL layers at a chosen token position; save as .pt
  4. Layer-wise logistic regression probe -> plot accuracy across layers
  5. Compute mean-difference direction per layer
  6. Paper's direction selection: bypass/induce/kl scores -> plot and select best layer
  7. Alpha search: generate on harmful_val under each candidate alpha, score
     refusal + coherence with a judge model, pick lowest-refusal alpha among
     those that stay coherent (see alpha_judge_search.py) -> plot
  8. Save final steering vectors (additive M2.1 + angular M2.2)

Usage:
    # Full run (auto-selects layer via paper's algorithm):
    python scripts/refusal_misaligned.py --model google/gemma-3-4b-it \\
        --judge_model Qwen/Qwen2.5-0.5B-Instruct

    # Skip recomputing splits/activations on re-runs:
    python scripts/refusal_misaligned.py --model google/gemma-3-4b-it \\
        --judge_model Qwen/Qwen2.5-0.5B-Instruct \\
        --reload_splits --reload_activations

    # Override layer manually after inspecting plots:
    python scripts/refusal_misaligned.py --model google/gemma-3-4b-it \\
        --judge_model Qwen/Qwen2.5-0.5B-Instruct \\
        --reload_splits --reload_activations --layer 18 \\
        --alpha_grid -1 -5 -10 -20 -50
"""

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from alpha_judge_search import alpha_judge_search, plot_alpha_judge_search
from common import (
    DATA_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    get_decoder_layers,
    get_device,
    load_model_and_tokenizer,
    model_slug,
    save_angular_direction,
    save_direction,
)

PLOTS_DIR = PROJECT_ROOT / "plots"
ACTIVATIONS_DIR = DATA_DIR / "refusal" / "activations"
SPLITS_DIR = DATA_DIR / "refusal"

# Broad set of refusal-starting strings observed across LLMs.
# Used when --use_default_refusal_tokens is passed.
DEFAULT_REFUSAL_STRINGS = [
    "I can't", "I cannot", "I won't", "I will not",
    "I'm sorry", "I am sorry", "Sorry",
    "As an", "As a", "Unfortunately",
]


def get_default_refusal_token_ids(tokenizer):
    """Convert DEFAULT_REFUSAL_STRINGS to token IDs (first token of each string)."""
    ids = set()
    for s in DEFAULT_REFUSAL_STRINGS:
        toks = tokenizer.encode(s, add_special_tokens=False)
        if toks:
            ids.add(toks[0])
    return sorted(ids)


# ---------------------------------------------------------------------------
# Refusal metric (Section B of the paper)
# ---------------------------------------------------------------------------

def _get_next_token_probs(model, tokenizer, prompt, enable_thinking=False):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.softmax(logits[0, -1, :].float(), dim=-1)


def detect_refusal_tokens(model, tokenizer, harmful_prompts, n_detect=12, top_k=20, enable_thinking=False):
    """Average next-token probs over n_detect harmful prompts; return top_k token ids."""
    probs_sum = None
    for prompt in tqdm(harmful_prompts[:n_detect], desc="detecting refusal tokens"):
        p = _get_next_token_probs(model, tokenizer, prompt, enable_thinking=enable_thinking)
        probs_sum = p if probs_sum is None else probs_sum + p
    avg_probs = probs_sum / n_detect
    top_ids = avg_probs.topk(top_k).indices.tolist()
    top_tokens = [(tokenizer.decode([tid]), tid, avg_probs[tid].item()) for tid in top_ids]
    return top_ids, top_tokens


def compute_refusal_metric(model, tokenizer, prompt, refusal_token_ids, hook_fn=None, layer_idx=None, enable_thinking=False):
    """logit(P_refusal) where P_refusal = sum of probs of refusal token ids."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    handles = []
    if hook_fn is not None and layer_idx is not None:
        layers = get_decoder_layers(model)
        handles.append(layers[layer_idx].register_forward_hook(hook_fn))

    try:
        with torch.no_grad():
            logits = model(**inputs).logits
    finally:
        for h in handles:
            h.remove()

    probs = torch.softmax(logits[0, -1, :].float(), dim=-1)
    token_ids_t = torch.tensor(refusal_token_ids, device=probs.device)
    p_refusal = probs[token_ids_t].sum().item()
    p_refusal = max(min(p_refusal, 1 - 1e-7), 1e-7)
    return math.log(p_refusal / (1 - p_refusal))


# ---------------------------------------------------------------------------
# Data: loading, filtering, splitting
# ---------------------------------------------------------------------------

def get_harmful_prompts(n):
    ds = load_dataset("walledai/AdvBench", split="train")
    return list(ds["prompt"][:n])


def get_harmless_prompts(n, seed=0):
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.filter(lambda ex: ex["input"] == "")
    indices = random.Random(seed).sample(range(len(ds)), n)
    return [ds[i]["instruction"] for i in indices]


def get_or_fetch_raw_harmful_pool(n):
    """Same raw candidate pool regardless of model (no filtering/model-specific
    scoring happens here yet), so this is cached once and shared across every
    model's run -- unlike the filtered splits/scores, which are model-specific."""
    path = SPLITS_DIR / f"raw_harmful_pool_{n}.json"
    if path.exists():
        return json.load(open(path))
    prompts = get_harmful_prompts(n)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(prompts, f)
    return prompts


def get_or_fetch_raw_harmless_pool(n, seed):
    path = SPLITS_DIR / f"raw_harmless_pool_{n}_{seed}.json"
    if path.exists():
        return json.load(open(path))
    prompts = get_harmless_prompts(n, seed=seed)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(prompts, f)
    return prompts


def filter_prompts(model, tokenizer, harmful_raw, harmless_raw, refusal_token_ids, enable_thinking=False):
    """Keep harmful where model refuses (metric>0); harmless where it doesn't (metric<0).

    Also returns the raw per-prompt scores (for the refusal-metric distribution plot).
    """
    harmful_scores = [
        compute_refusal_metric(model, tokenizer, p, refusal_token_ids, enable_thinking=enable_thinking)
        for p in tqdm(harmful_raw, desc="scoring harmful")
    ]
    harmless_scores = [
        compute_refusal_metric(model, tokenizer, p, refusal_token_ids, enable_thinking=enable_thinking)
        for p in tqdm(harmless_raw, desc="scoring harmless")
    ]
    harmful_filtered = [p for p, s in zip(harmful_raw, harmful_scores) if s > 0]
    harmless_filtered = [p for p, s in zip(harmless_raw, harmless_scores) if s < 0]
    return harmful_filtered, harmless_filtered, harmful_scores, harmless_scores


def plot_refusal_metric_distribution(harmful_scores, harmless_scores, out_dir, model=None):
    """Reproduces the paper's refusal-metric histogram (Fig. 10): harmful vs. harmless."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(harmless_scores, bins=30, color="tab:blue", alpha=0.6, label="harmless")
    ax.hist(harmful_scores, bins=30, color="tab:red", alpha=0.6, label="harmful")
    ax.set_xlabel("Refusal metric")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Refusal metric{f' -- {model}' if model else ''}")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "refusal_metric_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved refusal metric distribution plot to {path}")


def make_splits(harmful, harmless, n_train, n_val, seed=42):
    rng = random.Random(seed)
    rng.shuffle(harmful)
    rng.shuffle(harmless)
    return {
        "harmful_train": harmful[:n_train],
        "harmful_val": harmful[n_train:n_train + n_val],
        "harmless_train": harmless[:n_train],
        "harmless_val": harmless[n_train:n_train + n_val],
    }


def save_splits(splits, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, prompts in splits.items():
        with open(out_dir / f"{name}.json", "w") as f:
            json.dump(prompts, f, indent=2)


def load_splits(out_dir):
    out_dir = Path(out_dir)
    return {
        name: json.load(open(out_dir / f"{name}.json"))
        for name in ["harmful_train", "harmful_val", "harmless_train", "harmless_val"]
    }


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def capture_all_layers(model, tokenizer, prompt, token_pos=-1, enable_thinking=False):
    """Return [n_layers, hidden_dim] of residual-stream activations at token_pos."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    layers = get_decoder_layers(model)
    n_layers = len(layers)
    captured = [None] * n_layers

    def make_hook(i):
        def hook(_module, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            captured[i] = hidden[0, token_pos, :].detach().float().cpu()
        return hook

    handles = [layers[i].register_forward_hook(make_hook(i)) for i in range(n_layers)]
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in handles:
            h.remove()

    return torch.stack(captured)  # [n_layers, hidden_dim]


def extract_activations(model, tokenizer, prompts, token_pos, desc="extracting", enable_thinking=False):
    """Returns [n_prompts, n_layers, hidden_dim]."""
    acts = [
        capture_all_layers(model, tokenizer, p, token_pos, enable_thinking=enable_thinking)
        for p in tqdm(prompts, desc=desc)
    ]
    return torch.stack(acts)


def save_activations(harmful_acts, harmless_acts, out_dir, split_name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(harmful_acts, out_dir / f"harmful_{split_name}.pt")
    torch.save(harmless_acts, out_dir / f"harmless_{split_name}.pt")
    print(f"Saved {split_name} activations to {out_dir} (shape: {tuple(harmful_acts.shape)})")


def load_activations(out_dir, split_name):
    out_dir = Path(out_dir)
    return (
        torch.load(out_dir / f"harmful_{split_name}.pt", map_location="cpu"),
        torch.load(out_dir / f"harmless_{split_name}.pt", map_location="cpu"),
    )


# ---------------------------------------------------------------------------
# Layer-wise logistic regression probe
# ---------------------------------------------------------------------------

def run_layer_probing(train_harmful, train_harmless, val_harmful, val_harmless):
    """Returns list of val accuracy per layer. Inputs: [n_prompts, n_layers, hidden_dim]."""
    n_layers = train_harmful.shape[1]

    X_train = torch.cat([train_harmful, train_harmless], dim=0).numpy()
    y_train = [1] * len(train_harmful) + [0] * len(train_harmless)
    X_val = torch.cat([val_harmful, val_harmless], dim=0).numpy()
    y_val = [1] * len(val_harmful) + [0] * len(val_harmless)

    accuracies = []
    for l in tqdm(range(n_layers), desc="probing layers"):
        scaler = StandardScaler()
        X_tr_l = scaler.fit_transform(X_train[:, l, :])
        X_v_l = scaler.transform(X_val[:, l, :])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_tr_l, y_train)
        accuracies.append(clf.score(X_v_l, y_val))

    return accuracies


def plot_probe_accuracy(accuracies, out_dir, model=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = int(torch.tensor(accuracies).argmax())
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(accuracies)), accuracies, marker="o", markersize=3)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
    ax.axvline(best, color="red", linestyle="--", label=f"best layer {best} ({accuracies[best]:.2f})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Val accuracy")
    ax.set_title(f"Layer-wise probe accuracy (harmful vs. harmless){f' -- {model}' if model else ''}")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "probe_accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved probe accuracy plot to {path}")


# ---------------------------------------------------------------------------
# Mean-difference directions
# ---------------------------------------------------------------------------

def compute_directions(harmful_acts, harmless_acts):
    """Per-layer mean-difference direction, in two forms.

    Args:
        harmful_acts:  [n_harmful, n_layers, hidden_dim]
        harmless_acts: [n_harmless, n_layers, hidden_dim]
    Returns:
        unit_directions: [n_layers, hidden_dim], L2-normalized. Ablation's
            projection formula (h - (h.d)d) requires a unit vector, so this is
            what bypass/KL/M2.2 use.
        raw_directions: [n_layers, hidden_dim], unnormalized. Additive steering
            (induce score, alpha search, M2.1) uses this instead of the unit
            vector, so the injected magnitude reflects the real separation
            between harmful/harmless activations rather than an arbitrary
            length-1 nudge that's negligible next to real activation norms.
    """
    diff = harmful_acts.mean(dim=0) - harmless_acts.mean(dim=0)  # [n_layers, hidden_dim]
    norms = diff.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    unit_directions = diff / norms
    return unit_directions, diff


# ---------------------------------------------------------------------------
# Steering hooks
# ---------------------------------------------------------------------------

def make_ablation_hook(direction):
    """h' = h - (h*d)d  (directional ablation)."""
    d = direction.float()
    def hook(_module, _inp, out):
        is_tuple = isinstance(out, tuple)
        hidden = out[0] if is_tuple else out
        dv = d.to(hidden.dtype).to(hidden.device)
        proj = (hidden @ dv).unsqueeze(-1) * dv
        hidden = hidden - proj
        return (hidden, *out[1:]) if is_tuple else hidden
    return hook


def make_addition_hook(direction, coef):
    """h' = h + coef * d  (activation addition)."""
    d = direction.float()
    def hook(_module, _inp, out):
        is_tuple = isinstance(out, tuple)
        hidden = out[0] if is_tuple else out
        dv = d.to(hidden.dtype).to(hidden.device)
        hidden = hidden + coef * dv
        return (hidden, *out[1:]) if is_tuple else hidden
    return hook


# ---------------------------------------------------------------------------
# Paper's direction selection scores (Section C)
# ---------------------------------------------------------------------------

def _score_prompts(model, tokenizer, prompts, refusal_token_ids, hook_fn=None, layer_idx=None, enable_thinking=False):
    return [
        compute_refusal_metric(model, tokenizer, p, refusal_token_ids, hook_fn, layer_idx, enable_thinking=enable_thinking)
        for p in prompts
    ]


def compute_bypass_score(model, tokenizer, harmful_val, direction, refusal_token_ids, enable_thinking=False):
    """Mean refusal_metric on harmful prompts under directional ablation applied at every layer.

    Ablation must be applied everywhere the direction is written to the residual
    stream, not just at the layer it was extracted from -- otherwise downstream
    layers just reintroduce the refusal signal. Lower = better bypass.
    """
    layers = get_decoder_layers(model)
    ablation_hook = make_ablation_hook(direction)
    handles = [l.register_forward_hook(ablation_hook) for l in layers]
    try:
        scores = [
            compute_refusal_metric(model, tokenizer, p, refusal_token_ids, enable_thinking=enable_thinking)
            for p in harmful_val
        ]
    finally:
        for h in handles:
            h.remove()
    return sum(scores) / len(scores)


def compute_induce_score(model, tokenizer, harmless_val, direction, layer_idx, refusal_token_ids, enable_thinking=False):
    """Mean refusal_metric on harmless prompts under activation addition. Higher = better induction."""
    scores = _score_prompts(model, tokenizer, harmless_val, refusal_token_ids,
                            make_addition_hook(direction, 1.0), layer_idx, enable_thinking=enable_thinking)
    return sum(scores) / len(scores)


def compute_kl_score(model, tokenizer, harmless_val, direction, enable_thinking=False):
    """Mean KL(clean || ablated) on harmless prompts, ablating at every layer. Lower = less disruption."""
    layers = get_decoder_layers(model)
    ablation_hook = make_ablation_hook(direction)
    kl_divs = []
    for prompt in harmless_val:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits_clean = model(**inputs).logits[0, -1, :].float()
        handles = [l.register_forward_hook(ablation_hook) for l in layers]
        try:
            with torch.no_grad():
                logits_ablated = model(**inputs).logits[0, -1, :].float()
        finally:
            for h in handles:
                h.remove()
        p = torch.softmax(logits_clean, dim=-1)
        q = torch.softmax(logits_ablated, dim=-1)
        kl_divs.append((p * (torch.log(p + 1e-10) - torch.log(q + 1e-10))).sum().item())
    return sum(kl_divs) / len(kl_divs)


def select_best_direction(model, tokenizer, harmful_val, harmless_val,
                          directions, raw_directions, refusal_token_ids, max_layer_frac=0.9,
                          enable_thinking=False):
    """Evaluate bypass/induce/kl at every layer; select per paper's algorithm."""
    n_layers = directions.shape[0]
    layer_cap = int(max_layer_frac * n_layers)
    bypass_scores, induce_scores, kl_scores = [], [], []

    for l in tqdm(range(n_layers), desc="scoring directions"):
        d = directions[l]
        bypass = compute_bypass_score(model, tokenizer, harmful_val, d, refusal_token_ids, enable_thinking=enable_thinking)
        induce = compute_induce_score(model, tokenizer, harmless_val, raw_directions[l], l, refusal_token_ids, enable_thinking=enable_thinking)
        kl = compute_kl_score(model, tokenizer, harmless_val, d, enable_thinking=enable_thinking)
        bypass_scores.append(bypass)
        induce_scores.append(induce)
        kl_scores.append(kl)
        print(f"  layer {l:3d}: bypass={bypass:.3f}  induce={induce:.3f}  kl={kl:.4f}")

    best_layer, best_bypass = None, float("inf")
    for l in range(layer_cap):
        if induce_scores[l] > 0 and kl_scores[l] < 0.25 and bypass_scores[l] < best_bypass:
            best_bypass = bypass_scores[l]
            best_layer = l

    return best_layer, bypass_scores, induce_scores, kl_scores


def plot_direction_scores(bypass_scores, induce_scores, kl_scores, best_layer, out_dir, model=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = range(len(bypass_scores))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    specs = [
        (bypass_scores, "Bypass score (lower = better)", "tab:blue"),
        (induce_scores, "Induce score (higher = better)", "tab:orange"),
        (kl_scores,     "KL score (lower = less disruption)", "tab:green"),
    ]
    for ax, (scores, title, color) in zip(axes, specs):
        ax.plot(layers, scores, color=color)
        if best_layer is not None:
            ax.axvline(best_layer, color="red", linestyle="--", label=f"layer {best_layer}")
            ax.legend()
        ax.set_xlabel("Layer")
        ax.set_title(title)
    if model:
        fig.suptitle(model)
    plt.tight_layout()
    path = out_dir / "direction_selection.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved direction selection plot to {path}")


# ---------------------------------------------------------------------------
# Alpha grid search
# ---------------------------------------------------------------------------

def alpha_grid_search(model, tokenizer, harmful_val, direction, layer_idx, alpha_values, refusal_token_ids,
                      enable_thinking=False):
    """For each alpha, measure mean bypass_score on harmful_val with additive steering."""
    results = {}
    for alpha in tqdm(alpha_values, desc="alpha grid search"):
        scores = _score_prompts(model, tokenizer, harmful_val, refusal_token_ids,
                                make_addition_hook(direction, alpha), layer_idx, enable_thinking=enable_thinking)
        mean_score = sum(scores) / len(scores)
        results[alpha] = mean_score
        print(f"  alpha={alpha:+.3f}: mean bypass_score={mean_score:.3f}")
    return results


def plot_alpha_search(alpha_results, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    alphas = sorted(alpha_results.keys())
    scores = [alpha_results[a] for a in alphas]
    best_alpha = min(alpha_results, key=alpha_results.get)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(alphas, scores, marker="o")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5, label="refusal threshold")
    ax.axvline(best_alpha, color="red", linestyle="--", label=f"best alpha={best_alpha}")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Mean bypass score on val harmful")
    ax.set_title("Alpha grid search (lower = stronger refusal bypass)")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "alpha_search.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved alpha search plot to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    model,
    n_train=128,
    n_val=32,
    seed=42,
    n_raw_pool=500,
    token_pos=-1,
    reload_activations=False,
    reload_splits=False,
    use_default_refusal_tokens=False,
    n_detect_refusal=12,
    top_k_refusal=20,
    layer=None,
    alpha_grid=None,
    angular_coef=0.0,
    enable_thinking=False,
    judge_model=None,
    coherence_threshold=6.0,
    alpha_search_n_prompts=16,
    alpha_search_max_new_tokens=128,
    fixed_alpha=None,
    default_splits=False,
    base_model=None,
):
    """Core logic, callable directly (e.g. from modal/modal_app.py) without going through argparse."""
    if alpha_grid is None:
        alpha_grid = [-0.5, -1.0, -1.5, -2.0, -5.0]
    if default_splits and base_model is None:
        raise ValueError(
            "base_model is required when default_splits is set -- it's whose already-filtered "
            "splits get reused (e.g. an ablated/finetuned model whose own refusal-metric filtering "
            "fails or comes up empty should reuse the original base model's harmful/harmless split, "
            "rather than attempt to filter itself)."
        )
    if fixed_alpha is None and judge_model is None:
        raise ValueError(
            "judge_model is required unless --fixed_alpha is set: Step 7 generates real responses "
            "under each candidate alpha and scores them with this model for refusal + coherence, "
            "instead of just looking at next-token logits."
        )

    random.seed(seed)
    torch.manual_seed(seed)

    device = get_device()

    model_obj, tokenizer = load_model_and_tokenizer(model, device=device)
    slug = model_slug(model)
    model_activations_dir = ACTIVATIONS_DIR / slug
    model_activations_dir.mkdir(parents=True, exist_ok=True)
    model_plots_dir = PLOTS_DIR / slug
    model_plots_dir.mkdir(parents=True, exist_ok=True)
    # Refusal tokens, splits, and refusal-metric scores are all specific to
    # THIS model's own refusal behavior -- keeping them under a shared,
    # non-model-specific directory would mean one model's run silently
    # overwrites another's, and a later --reload_splits/--use_default_refusal_tokens
    # run could load the wrong model's cached data entirely.
    model_splits_dir = SPLITS_DIR / slug
    model_splits_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Detect / load refusal tokens ---
    print("\n=== Step 1: Refusal tokens ===")
    raw_harmful_pool = get_or_fetch_raw_harmful_pool(n_raw_pool)
    tokens_path = model_splits_dir / "refusal_tokens.json"

    if use_default_refusal_tokens:
        refusal_token_ids = get_default_refusal_token_ids(tokenizer)
        token_info = [{"token": tokenizer.decode([tid]), "id": tid, "source": "default"}
                      for tid in refusal_token_ids]
        print(f"Using default refusal token set ({len(refusal_token_ids)} tokens): "
              f"{[tokenizer.decode([tid]) for tid in refusal_token_ids]}")
    else:
        refusal_token_ids, refusal_token_info = detect_refusal_tokens(
            model_obj, tokenizer, raw_harmful_pool,
            n_detect=n_detect_refusal, top_k=top_k_refusal, enable_thinking=enable_thinking,
        )
        token_info = [{"token": t, "id": tid, "prob": p} for t, tid, p in refusal_token_info]
        print(f"Auto-detected refusal tokens: {[(t, f'{p:.3f}') for t, tid, p in refusal_token_info[:10]]}")

    with open(tokens_path, "w") as f:
        json.dump(token_info, f, indent=2)
    print(f"Saved to {tokens_path}")

    # --- Step 2: Build / load splits ---
    scores_path = model_splits_dir / "refusal_scores.json"
    if default_splits:
        # For models whose own refusal-metric filtering fails or comes up
        # empty (e.g. an ablated/compliance-tuned model that rarely refuses,
        # so nothing clears the harmful-prompt filter) -- reuse base_model's
        # already-filtered split wholesale instead of attempting to filter
        # this model's own prompts. The distribution plot, however, is still
        # scored on THIS model's own raw pool (same shared 500-prompt
        # harmful/harmless pool the non-default_splits path would use, not
        # base_model's smaller already-filtered split) so it's comparable
        # across models regardless of which one happened to supply the
        # actual splits -- purely diagnostic, these scores never decide
        # split membership here, so they can't reproduce the original
        # empty-split failure.
        base_splits_dir = SPLITS_DIR / model_slug(base_model)
        if not (base_splits_dir / "harmful_train.json").exists():
            raise FileNotFoundError(
                f"No splits found for base_model at {base_splits_dir}. Run scripts/refusal_misaligned.py "
                f"--model {base_model} first to produce them."
            )
        print(f"\n=== Step 2: Using {base_model}'s splits (default_splits) ===")
        splits = load_splits(base_splits_dir)
        print(f"Scoring this model on its own raw pool (n={n_raw_pool}) for the distribution plot "
              f"only (not used to filter/decide split membership -- {base_model}'s splits are used for that)...")
        raw_harmless_pool = get_or_fetch_raw_harmless_pool(n_raw_pool, seed)
        harmful_scores = [
            compute_refusal_metric(model_obj, tokenizer, p, refusal_token_ids, enable_thinking=enable_thinking)
            for p in tqdm(raw_harmful_pool, desc="scoring harmful (diagnostic only)")
        ]
        harmless_scores = [
            compute_refusal_metric(model_obj, tokenizer, p, refusal_token_ids, enable_thinking=enable_thinking)
            for p in tqdm(raw_harmless_pool, desc="scoring harmless (diagnostic only)")
        ]
        plot_refusal_metric_distribution(harmful_scores, harmless_scores, model_plots_dir, model=model)
        with open(scores_path, "w") as f:
            json.dump({"harmful_scores": harmful_scores, "harmless_scores": harmless_scores}, f)
    elif reload_splits and (model_splits_dir / "harmful_train.json").exists():
        print("\n=== Step 2: Loading saved splits ===")
        splits = load_splits(model_splits_dir)
        if scores_path.exists():
            scores = json.load(open(scores_path))
            plot_refusal_metric_distribution(scores["harmful_scores"], scores["harmless_scores"], model_plots_dir, model=model)
        else:
            print(f"No cached scores at {scores_path} (from a run before this was added) -- "
                  f"skipping refusal_metric_distribution.png. Re-run without --reload_splits once to cache them.")
    else:
        print("\n=== Step 2: Filtering and splitting prompts ===")
        raw_harmless_pool = get_or_fetch_raw_harmless_pool(n_raw_pool, seed)
        harmful_filtered, harmless_filtered, harmful_scores, harmless_scores = filter_prompts(
            model_obj, tokenizer, raw_harmful_pool, raw_harmless_pool, refusal_token_ids, enable_thinking=enable_thinking
        )
        print(f"After filtering: {len(harmful_filtered)} harmful, {len(harmless_filtered)} harmless")
        plot_refusal_metric_distribution(harmful_scores, harmless_scores, model_plots_dir, model=model)
        with open(scores_path, "w") as f:
            json.dump({"harmful_scores": harmful_scores, "harmless_scores": harmless_scores}, f)
        splits = make_splits(harmful_filtered, harmless_filtered, n_train, n_val, seed=seed)
        save_splits(splits, model_splits_dir)
        print(f"Saved splits to {model_splits_dir}")

    harmful_train  = splits["harmful_train"]
    harmful_val    = splits["harmful_val"]
    harmless_train = splits["harmless_train"]
    harmless_val   = splits["harmless_val"]

    for name, split in [("harmful_train", harmful_train), ("harmful_val", harmful_val),
                        ("harmless_train", harmless_train), ("harmless_val", harmless_val)]:
        if len(split) == 0:
            raise RuntimeError(
                f"{name} is empty after filtering -- can't extract a refusal direction with no examples. "
                f"If this is a harmful_* split, the model likely never (or rarely) refuses on this dataset "
                f"under the current refusal_token_ids (check the 'After filtering: N harmful, M harmless' "
                f"line above); if harmless_*, the model may over-refuse instead. Either way this reflects "
                f"actual model behavior, not a bug -- try --use_default_refusal_tokens off (or on, if it "
                f"was off) to see if auto-detected tokens pick up a real signal, or increase --n_raw_pool."
            )

    # --- Step 3: Extract / load activations ---
    if reload_activations and (model_activations_dir / "harmful_train.pt").exists():
        print("\n=== Step 3: Loading saved activations ===")
        train_harmful_acts,  train_harmless_acts = load_activations(model_activations_dir, "train")
        val_harmful_acts,    val_harmless_acts   = load_activations(model_activations_dir, "val")
    else:
        print("\n=== Step 3: Extracting activations from all layers ===")
        train_harmful_acts  = extract_activations(model_obj, tokenizer, harmful_train,  token_pos, "train harmful", enable_thinking=enable_thinking)
        train_harmless_acts = extract_activations(model_obj, tokenizer, harmless_train, token_pos, "train harmless", enable_thinking=enable_thinking)
        val_harmful_acts    = extract_activations(model_obj, tokenizer, harmful_val,    token_pos, "val harmful", enable_thinking=enable_thinking)
        val_harmless_acts   = extract_activations(model_obj, tokenizer, harmless_val,   token_pos, "val harmless", enable_thinking=enable_thinking)
        save_activations(train_harmful_acts, train_harmless_acts, model_activations_dir, "train")
        save_activations(val_harmful_acts,   val_harmless_acts,   model_activations_dir, "val")

    # --- Step 4: Layer-wise logistic regression ---
    print("\n=== Step 4: Layer-wise logistic regression probe ===")
    probe_accuracies = run_layer_probing(
        train_harmful_acts, train_harmless_acts, val_harmful_acts, val_harmless_acts
    )
    plot_probe_accuracy(probe_accuracies, model_plots_dir, model=model)
    best_probe_layer = int(torch.tensor(probe_accuracies).argmax().item())
    print(f"Best probe layer: {best_probe_layer} (acc={probe_accuracies[best_probe_layer]:.3f})")

    # --- Step 5: Compute mean-difference directions per layer ---
    print("\n=== Step 5: Computing mean-difference directions ===")
    directions, raw_directions = compute_directions(train_harmful_acts, train_harmless_acts)  # [n_layers, hidden_dim] each

    # --- Step 6: Direction selection ---
    if layer is None:
        print("\n=== Step 6: Paper direction selection (bypass / induce / kl) ===")
        best_layer, bypass_scores, induce_scores, kl_scores = select_best_direction(
            model_obj, tokenizer, harmful_val, harmless_val, directions, raw_directions, refusal_token_ids,
            enable_thinking=enable_thinking,
        )
        plot_direction_scores(bypass_scores, induce_scores, kl_scores, best_layer, model_plots_dir, model=model)
        if best_layer is None:
            print("WARNING: No layer passed all filters; falling back to best probe layer.")
            best_layer = best_probe_layer
        print(f"Selected layer: {best_layer}")

        scores_json_path = model_plots_dir / "direction_scores.json"
        with open(scores_json_path, "w") as f:
            json.dump({
                "model": model,
                "best_layer": best_layer,
                "bypass_scores": bypass_scores,
                "induce_scores": induce_scores,
                "kl_scores": kl_scores,
            }, f, indent=2)
        print(f"Saved direction scores to {scores_json_path}")
    else:
        best_layer = layer
        print(f"\n=== Step 6: Using manually specified layer {best_layer} ===")

    best_direction = directions[best_layer]
    best_raw_direction = raw_directions[best_layer]

    # --- Step 7: Alpha search via judge-scored generation ---
    if fixed_alpha is not None:
        print("\n=== Step 7: Skipped (using fixed alpha) ===")
        best_alpha = fixed_alpha
        print(f"Using fixed alpha: {best_alpha}")
    else:
        # Generates actual responses under each candidate alpha (on harmful_val,
        # since we're steering AGAINST refusal -- nothing to bypass on harmless
        # prompts) and scores them with a separate judge model on refusal +
        # coherence, instead of just reading next-token logits like
        # alpha_grid_search does.
        print("\n=== Step 7: Alpha search (judge-scored generation) ===")
        print(f"Loading judge model: {judge_model}")
        judge_model_obj, judge_tokenizer = load_model_and_tokenizer(judge_model, device=device)
        try:
            alpha_judge_results, best_alpha = alpha_judge_search(
                model_obj, tokenizer, judge_model_obj, judge_tokenizer,
                harmful_val, best_raw_direction, best_layer, alpha_grid,
                n_prompts=alpha_search_n_prompts, max_new_tokens=alpha_search_max_new_tokens,
                coherence_threshold=coherence_threshold, enable_thinking=enable_thinking,
            )
        finally:
            del judge_model_obj, judge_tokenizer
            if device == "cuda":
                torch.cuda.empty_cache()

        plot_alpha_judge_search(alpha_judge_results, best_alpha, model_plots_dir, model=model)
        if best_alpha is None:
            print(f"WARNING: No alpha cleared coherence_threshold={coherence_threshold}; "
                  f"falling back to lowest-refusal alpha regardless of coherence.")
            best_alpha = min(alpha_judge_results, key=lambda a: alpha_judge_results[a]["refusal"])
        print(f"Best alpha: {best_alpha} "
              f"(refusal={alpha_judge_results[best_alpha]['refusal']:.2f}, "
              f"coherence={alpha_judge_results[best_alpha]['coherence']:.2f})")

    # --- Step 8: Save steering vectors ---
    print("\n=== Step 8: Saving steering vectors ===")
    n_layers = directions.shape[0]
    # Additive steering (M2.1) is a targeted nudge -- applied at the single
    # layer it was tuned at, matching alpha_grid_search above. It uses the raw
    # (unnormalized) mean-difference vector, not the unit direction, so the
    # injected magnitude reflects the real separation between harmful/harmless
    # activations instead of an arbitrary length-1 nudge.
    # Angular/ablation steering (M2.2) must remove the direction everywhere
    # it's written to the residual stream, matching compute_bypass_score above,
    # otherwise downstream layers just reintroduce the refusal signal; its
    # projection formula requires the unit direction.
    additive_layers = [best_layer]
    ablation_layers = list(range(n_layers))
    out_root = MODELS_DIR / slug

    additive_path = out_root / "M2.1_steer_against_refusal_additive" / "direction.pt"
    angular_path = out_root / "M2.2_steer_against_refusal_angular" / "direction.pt"
    save_direction(best_raw_direction, best_alpha, "additive", additive_layers, additive_path)

    # True angular steering needs a 2D plane, not a single direction: b1 is the
    # selected layer's unit direction, b2 is an orthogonal direction found via
    # PCA over the per-layer directions (how the direction vector's endpoint
    # varies across depth), so steering can rotate within that plane instead
    # of just pinning a single projection value.
    b1 = best_direction
    X = directions - directions.mean(dim=0)
    _, _, V = torch.pca_lowrank(X, q=3)
    b2 = V[:, 1]
    b2 = b2 - torch.dot(b2, b1) * b1
    b2 = b2 / b2.norm()

    save_angular_direction(b1, b2, angular_coef, ablation_layers, angular_path)

    print(f"Saved M2.1 additive (alpha={best_alpha}, layers={additive_layers})")
    print(f"Saved M2.2 angular (theta_deg={angular_coef}, layers={ablation_layers})")
    print(f"All plots saved to: {model_plots_dir}")
    print("Done.")
    return additive_path, angular_path


def main():
    parser = argparse.ArgumentParser(description="M2: probe and steer against the refusal direction.")
    # Data
    parser.add_argument("--model", required=True)
    parser.add_argument("--n_train", type=int, default=128)
    parser.add_argument("--n_val", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_raw_pool", type=int, default=500,
                        help="Candidate prompts to load before filtering")
    # Activation extraction
    parser.add_argument("--token_pos", type=int, default=-1,
                        help="Token position to probe (-1=last, -2=second-to-last, ...)")
    parser.add_argument("--reload_activations", action="store_true",
                        help="Load saved activations from disk instead of recomputing")
    parser.add_argument("--reload_splits", action="store_true",
                        help="Load saved splits from disk instead of recomputing")
    parser.add_argument("--default_splits", action="store_true",
                        help="Reuse --base_model's already-filtered splits wholesale instead of "
                             "filtering this model's own prompts -- for models whose own refusal-metric "
                             "filtering fails or comes up empty (e.g. a heavily ablated/compliance-tuned "
                             "model that rarely refuses). Still scores this model on the reused prompts "
                             "for the distribution plot, but only as a diagnostic, not to decide membership.")
    parser.add_argument("--base_model", default=None,
                        help="Whose splits to reuse when --default_splits is set (required if so)")
    # Refusal token detection
    parser.add_argument("--use_default_refusal_tokens", action="store_true",
                        help="Use the hardcoded DEFAULT_REFUSAL_STRINGS instead of auto-detecting from the model")
    parser.add_argument("--n_detect_refusal", type=int, default=12,
                        help="Number of harmful prompts used to detect refusal tokens (auto-detect only)")
    parser.add_argument("--top_k_refusal", type=int, default=20,
                        help="How many top tokens to treat as refusal tokens (auto-detect only)")
    # Direction / layer selection
    parser.add_argument("--layer", type=int, default=None,
                        help="Manually specify layer (skips paper direction selection)")
    # Alpha
    parser.add_argument("--alpha_grid", type=float, nargs="+",
                        default=[-0.5, -1.0, -1.5, -2.0, -5.0,],
                        help="Alpha values for additive steering grid search (negative pushes away from refusal)")
    parser.add_argument("--angular_coef", type=float, default=0.0,
                        help="Angular steering target angle theta (degrees) between b1 (refusal "
                             "direction) and b2 (orthogonal PCA direction); 0 = full alignment with b1")
    # Judge-based alpha search (Step 7)
    parser.add_argument("--judge_model", default=None,
                        help="Model used to score generated responses for refusal + coherence "
                             "during alpha search (e.g. a small local model). Required unless "
                             "--fixed_alpha is set.")
    parser.add_argument("--fixed_alpha", type=float, default=None,
                        help="Skip the judge-scored alpha search entirely and use this alpha "
                             "directly for M2.1 (e.g. -1). judge_model not needed if set.")
    parser.add_argument("--coherence_threshold", type=float, default=6.0,
                        help="Minimum mean judge coherence (0-10) an alpha must clear to be "
                             "eligible for selection; among eligible alphas, lowest refusal wins")
    parser.add_argument("--alpha_search_n_prompts", type=int, default=16,
                        help="Number of harmful_val prompts to generate on per alpha")
    parser.add_argument("--alpha_search_max_new_tokens", type=int, default=128,
                        help="Max new tokens to generate per response during alpha search")
    # Chat template
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Enable thinking mode in the chat template (Qwen3-style models). "
                             "Default off, since the refusal metric probes the first response token "
                             "and thinking models otherwise emit <think> there instead of a refusal cue.")
    args = parser.parse_args()

    run(
        args.model,
        n_train=args.n_train,
        n_val=args.n_val,
        seed=args.seed,
        n_raw_pool=args.n_raw_pool,
        token_pos=args.token_pos,
        reload_activations=args.reload_activations,
        reload_splits=args.reload_splits,
        use_default_refusal_tokens=args.use_default_refusal_tokens,
        n_detect_refusal=args.n_detect_refusal,
        top_k_refusal=args.top_k_refusal,
        layer=args.layer,
        alpha_grid=args.alpha_grid,
        angular_coef=args.angular_coef,
        enable_thinking=args.enable_thinking,
        judge_model=args.judge_model,
        coherence_threshold=args.coherence_threshold,
        alpha_search_n_prompts=args.alpha_search_n_prompts,
        alpha_search_max_new_tokens=args.alpha_search_max_new_tokens,
        fixed_alpha=args.fixed_alpha,
        default_splits=args.default_splits,
        base_model=args.base_model,
    )


if __name__ == "__main__":
    main()
