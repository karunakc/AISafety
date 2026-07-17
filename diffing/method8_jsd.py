"""
Method 8: distribution-level divergence (JSD by default) between two
variants' next-token distributions, evaluated over a dataset of prompts.

Unlike methods 1-7 (which diff cached hidden-state activations or extracted
directions), this method diffs the two models' OUTPUT DISTRIBUTIONS
directly: for each prompt, variant_a greedily generates a continuation, then
BOTH variants are teacher-forced over the same (prompt + continuation) token
sequence so their per-token next-token distributions are directly comparable
position-by-position. Since both models are conditioned on the exact same
token history at every step, any divergence in the distribution over the
NEXT token is attributable to the models themselves, not to them having
already generated different text.

Reuses evaluations/eval_common.py::load_variant, so it compares ANY two
variants (base, M1, M2, M2.3, composites, ...) the same way run_eval.py does
-- Base vs LoRA, Base vs Steered Base, LoRA vs Steered LoRA, etc.

Usage:
    python diffing/method8_jsd.py --model Qwen/Qwen3.5-4B --variant_a base --variant_b M2 \\
        --categories harmful harmless --n_prompts 50

    # Compare two composites, computing several divergence metrics from the
    # same generation pass (one output JSON/plot pair per metric):
    python diffing/method8_jsd.py --model Qwen/Qwen3.5-4B --variant_a M1_good_medical_advice \\
        --variant_b M1_medical-M2 --metric jsd kl tv hellinger --n_prompts 30
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "evaluations"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval_common import load_variant, model_slug, remove_hooks  # noqa: E402
from jsd import METRICS  # noqa: E402
from refusal_misaligned import get_or_fetch_raw_harmful_pool, get_or_fetch_raw_harmless_pool  # noqa: E402
from teacher_forcing import generate_and_teacher_force  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_prompts(categories, n_prompts, seed=42, jailbreak_file=None):
    """Returns a list of (prompt, category) pairs, up to n_prompts per
    category. "harmful"/"harmless" reuse refusal_misaligned.py's own
    get_or_fetch_raw_*_pool -- same cache-if-present-else-fetch-from-HF
    behavior induce_refusal already relies on, so this works unmodified both
    locally (already cached under data/refusal/) and on Modal (fetches fresh,
    since Modal instances have internet unlike SLURM compute nodes).
    "jailbreak" has no such fetcher -- pass --jailbreak_file (a plain JSON
    list of prompt strings) to add one."""
    pools = {
        "harmful": lambda: get_or_fetch_raw_harmful_pool(n_prompts),
        "harmless": lambda: get_or_fetch_raw_harmless_pool(n_prompts, seed),
    }
    if jailbreak_file:
        pools["jailbreak"] = lambda: json.load(open(jailbreak_file))

    prompts = []
    for category in categories:
        if category not in pools:
            raise ValueError(f"No prompt pool for category {category!r} (available: {list(pools)}, "
                              f"or pass --jailbreak_file for a custom 'jailbreak' pool)")
        prompts.extend((p, category) for p in pools[category]()[:n_prompts])
    return prompts


@torch.no_grad()
def prompt_divergence(model_a, model_b, tokenizer, prompt, max_new_tokens=64, metrics=("jsd",)):
    """Greedily generate a continuation with model_a, then teacher-force both
    models over the shared (prompt + continuation) sequence (via
    teacher_forcing.generate_and_teacher_force) and compare their per-token
    next-token distributions over the generated span only (the prompt itself
    isn't "generated" by either model, so it's not a meaningful comparison
    point). Generation and both forward passes happen once regardless of how
    many metrics are requested -- they're all cheap postprocessing of the
    same pair of logit tensors. Returns a dict of {metric_name: 1D tensor},
    one value per generated token (shorter than max_new_tokens if generation
    hit EOS early; empty if generation produced 0 new tokens)."""
    tf = generate_and_teacher_force(model_a, model_b, tokenizer, prompt, max_new_tokens=max_new_tokens)
    if tf["gen_len"] == 0:
        return {metric: torch.empty(0) for metric in metrics}

    logits_a, logits_b = tf["logits_a"], tf["logits_b"]
    return {
        metric: METRICS[metric](logits_a, logits_b, reduction="none").squeeze(0).cpu()
        for metric in metrics
    }


def summarize(values):
    if not values:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None, "n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()), "median": float(np.median(arr)), "std": float(arr.std()),
        "min": float(arr.min()), "max": float(arr.max()), "n": int(arr.size),
    }


def run(model, variant_a, variant_b, categories=None, n_prompts=50, max_new_tokens=64, metrics=("jsd",),
        seed=42, jailbreak_file=None, label=None, output_dir=None, alpha_override=None):
    metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        raise ValueError(f"Unknown metric(s) {unknown}, expected any of {list(METRICS)}")
    categories = categories or ["harmful", "harmless"]
    prompts = load_prompts(categories, n_prompts, seed, jailbreak_file)

    model_a, tokenizer, handles_a = load_variant(model, variant_a, alpha_override=alpha_override)
    model_b, _, handles_b = load_variant(model, variant_b, alpha_override=alpha_override)
    try:
        per_prompt = {m: [] for m in metrics}
        per_token_curves = {m: [] for m in metrics}
        desc = f"{'+'.join(metrics)}({variant_a} || {variant_b})"
        for prompt, category in tqdm(prompts, desc=desc):
            token_divs = prompt_divergence(model_a, model_b, tokenizer, prompt,
                                            max_new_tokens=max_new_tokens, metrics=metrics)
            if token_divs[metrics[0]].numel() == 0:
                continue
            for m in metrics:
                token_div = token_divs[m]
                per_prompt[m].append({
                    "prompt": prompt, "category": category,
                    "mean": float(token_div.mean()), "per_token": token_div.tolist(),
                })
                per_token_curves[m].append(token_div)
    finally:
        remove_hooks(handles_a)
        remove_hooks(handles_b)

    results_dir = Path(output_dir) if output_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for m in metrics:
        mean_values = [p["mean"] for p in per_prompt[m]]
        summary = summarize(mean_values)
        by_category = {
            category: summarize([p["mean"] for p in per_prompt[m] if p["category"] == category])
            for category in categories
        }

        curves = per_token_curves[m]
        max_len = max((len(c) for c in curves), default=0)
        per_position_mean = [
            float(np.mean([c[i].item() for c in curves if i < len(c)])) for i in range(max_len)
        ]

        result = {
            "method": "jsd", "metric": m, "model": model, "variant_a": variant_a, "variant_b": variant_b,
            "alpha_override": alpha_override, "n_prompts": len(per_prompt[m]),
            "summary": summary, "by_category": by_category, "per_position_mean": per_position_mean,
            "per_prompt": per_prompt[m],
        }
        results[m] = result

        stem = label or f"{model_slug(model)}__{variant_a}_vs_{variant_b}"
        if len(metrics) > 1 or not label:
            stem += f"__{m}"
        if alpha_override is not None:
            stem += f"__alpha{alpha_override}"

        json_path = results_dir / f"{stem}.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved result to {json_path}")
        print(f"[{m}] summary: {summary}")
        print(f"[{m}] by_category: {by_category}")

        _plot_histogram(mean_values, results_dir / f"{stem}_hist.png", variant_a, variant_b, m)
        _plot_per_position(per_position_mean, results_dir / f"{stem}_per_token.png", variant_a, variant_b, m)

    return results if len(metrics) > 1 else results[metrics[0]]


def _plot_histogram(values, path, variant_a, variant_b, metric):
    if not values:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=30, color="#2a78d6", edgecolor="white")
    ax.set_xlabel(f"{metric.upper()} ({variant_a} || {variant_b}), mean per prompt")
    ax.set_ylabel("Prompts")
    ax.set_title(f"{metric.upper()} distribution: {variant_a} vs {variant_b}")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved histogram to {path}")


def _plot_per_position(per_position_mean, path, variant_a, variant_b, metric):
    if not per_position_mean:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(per_position_mean)), per_position_mean, marker="o", markersize=3, color="#e34948")
    ax.set_xlabel("Generated token position")
    ax.set_ylabel(f"Mean {metric.upper()}")
    ax.set_title(f"{metric.upper()} vs. generation position: {variant_a} vs {variant_b}")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved per-token plot to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Method 8: distribution-level divergence (JSD by default) between two variants."
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--variant_a", required=True, help="e.g. base (as understood by evaluations/eval_common.py::load_variant)")
    parser.add_argument("--variant_b", required=True)
    parser.add_argument("--categories", nargs="+", default=["harmful", "harmless"], help="Prompt pools to draw from")
    parser.add_argument("--jailbreak_file", default=None,
                         help="Optional JSON list of jailbreak prompts, loaded under category 'jailbreak' "
                              "(no such dataset ships in this repo by default)")
    parser.add_argument("--n_prompts", type=int, default=50, help="Prompts per category")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Generated tokens compared per prompt")
    parser.add_argument("--metric", nargs="+", default=["jsd"], choices=list(METRICS),
                         help="One or more divergence metrics to compute (default: jsd). Generation and the "
                              "forward passes only happen once per prompt regardless of how many are requested -- "
                              "each metric is just cheap postprocessing of the same logits -- and each metric gets "
                              "its own output JSON/plots.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the harmless pool fetch (matches this repo's cached raw_harmless_pool_500_42.json)")
    parser.add_argument("--label", default=None, help="Output filename stem under diffing/results/ (default: auto-generated)")
    parser.add_argument("--output_dir", default=None, help="Directory to save the result JSON/plots to (default: diffing/results/)")
    parser.add_argument("--alpha_override", type=float, default=None,
                         help="Override the steering coefficient MAGNITUDE for variant_a/variant_b, if either is a "
                              "steering variant (e.g. M2) or M1+M2 composite (default: each variant's own saved "
                              "coef). load_variant applies each variant's own sign, so this should always be given "
                              "as a positive magnitude -- e.g. 1.0 for M2, whose own coef is negative, yields an "
                              "effective coefficient of -1.0. Ignored by variants with no steering component "
                              "(base, M1, M1_*).")
    args = parser.parse_args()
    run(args.model, args.variant_a, args.variant_b, args.categories, args.n_prompts, args.max_new_tokens,
        args.metric, args.seed, args.jailbreak_file, args.label, args.output_dir, args.alpha_override)


if __name__ == "__main__":
    main()
