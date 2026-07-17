"""
Method 9: refusal-probability curves between two variants' next-token
distributions, evaluated over a dataset of prompts.

Same teacher-forcing setup as method8_jsd.py (variant_a greedily generates,
both variants are teacher-forced over the shared token sequence) but instead
of a general distributional divergence, this asks a targeted question: how
much probability mass does each variant put on BEGINNING a refusal, at every
decoding position? That answers "does variant_b specifically suppress
refusal, or does it just reweight the whole distribution generically" --
method8's JSD tells you THAT the distributions differ, this tells you
WHETHER refusal mass specifically is what differs.

Two refusal-probability estimators are computed (see diffing/refusal.py for
the full explanation of why they can disagree):
  - "exact": the true joint (chain-rule) probability of each configurable
    multi-token refusal phrase, via extra forward passes (subsampled every
    `--stride` positions to bound cost)
  - "token": a free, first-token-only baseline reusing the same logits
    method8 already computes, for comparison against "exact"

Usage:
    python diffing/method9_refusal_probability.py --model Qwen/Qwen3.5-4B \\
        --variant_a base --variant_b M2 --categories harmful --n_prompts 50

    # Custom refusal phrase list, denser exact-mode sampling:
    python diffing/method9_refusal_probability.py --model Qwen/Qwen3.5-4B \\
        --variant_a base --variant_b M2 --phrases "I can't" "I cannot" "I'm sorry" \\
        --stride 1
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "evaluations"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_common import load_variant, model_slug, remove_hooks  # noqa: E402

import plots  # noqa: E402
from curve_utils import per_position_mean_dense, per_position_mean_sparse  # noqa: E402
from method8_jsd import load_prompts, summarize  # noqa: E402
from refusal import (  # noqa: E402
    DEFAULT_REFUSAL_PHRASES,
    refusal_curve_stats,
    refusal_curves_for_model,
    tokenize_phrases,
)
from teacher_forcing import generate_and_teacher_force  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run(model, variant_a, variant_b, categories=None, n_prompts=50, max_new_tokens=64,
        phrases=None, stride=4, mode="both", seed=42, jailbreak_file=None, label=None,
        output_dir=None, alpha_override=None):
    if mode not in ("token", "exact", "both"):
        raise ValueError(f"mode must be 'token', 'exact', or 'both', got {mode!r}")
    categories = categories or ["harmful", "harmless"]
    phrases = list(phrases) if phrases else list(DEFAULT_REFUSAL_PHRASES)
    prompts = load_prompts(categories, n_prompts, seed, jailbreak_file)

    model_a, tokenizer, handles_a = load_variant(model, variant_a, alpha_override=alpha_override)
    model_b, _, handles_b = load_variant(model, variant_b, alpha_override=alpha_override)
    phrase_ids = tokenize_phrases(tokenizer, phrases)

    per_prompt = []
    dense_curves = {"a": [], "b": []}       # token-mode, for the per-position-mean plot
    sparse_curves = {"a": [], "b": []}      # exact-mode, (curve, positions) pairs
    headline_key = "exact_curve" if mode in ("exact", "both") else "token_curve"

    try:
        desc = f"refusal_prob({variant_a} || {variant_b})"
        for prompt, category in tqdm(prompts, desc=desc):
            tf = generate_and_teacher_force(model_a, model_b, tokenizer, prompt, max_new_tokens=max_new_tokens)
            if tf["gen_len"] == 0:
                continue
            start = tf["prefix_len"] - 1
            res_a = refusal_curves_for_model(model_a, tf["logits_a"][0], tf["full_ids"], start, tf["gen_len"],
                                              phrase_ids, stride, mode)
            res_b = refusal_curves_for_model(model_b, tf["logits_b"][0], tf["full_ids"], start, tf["gen_len"],
                                              phrase_ids, stride, mode)

            entry = {"prompt": prompt, "category": category, "gen_len": tf["gen_len"]}
            for name, res in (("a", res_a), ("b", res_b)):
                if "token_curve" in res:
                    entry[f"token_curve_{name}"] = res["token_curve"].tolist()
                    dense_curves[name].append(res["token_curve"])
                if "exact_curve" in res:
                    entry[f"exact_curve_{name}"] = res["exact_curve"].tolist()
                    entry[f"exact_positions_{name}"] = res["exact_positions"]
                    sparse_curves[name].append((res["exact_curve"], res["exact_positions"]))
                headline_curve = res.get(headline_key)
                headline_positions = res.get("exact_positions") if headline_key == "exact_curve" else None
                entry[f"stats_{name}"] = refusal_curve_stats(headline_curve, headline_positions) if headline_curve is not None else None
            per_prompt.append(entry)
    finally:
        remove_hooks(handles_a)
        remove_hooks(handles_b)

    results_dir = Path(output_dir) if output_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = label or f"{model_slug(model)}__{variant_a}_vs_{variant_b}__refusal"
    if alpha_override is not None:
        stem += f"__alpha{alpha_override}"

    summary = {
        "a": summarize([p["stats_a"]["mean"] for p in per_prompt if p["stats_a"] and p["stats_a"]["mean"] is not None]),
        "b": summarize([p["stats_b"]["mean"] for p in per_prompt if p["stats_b"] and p["stats_b"]["mean"] is not None]),
    }
    by_category = {
        name: {
            category: summarize([
                p[f"stats_{name}"]["mean"] for p in per_prompt
                if p["category"] == category and p[f"stats_{name}"] and p[f"stats_{name}"]["mean"] is not None
            ])
            for category in categories
        }
        for name in ("a", "b")
    }

    result = {
        "method": "refusal_probability", "model": model, "variant_a": variant_a, "variant_b": variant_b,
        "phrases": phrases, "mode": mode, "stride": stride, "alpha_override": alpha_override,
        "n_prompts": len(per_prompt), "headline_metric": headline_key,
        "summary": summary, "by_category": by_category, "per_prompt": per_prompt,
    }

    json_path = results_dir / f"{stem}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to {json_path}")
    print(f"summary: {summary}")
    print(f"by_category: {by_category}")

    def _plot_paired_curves(pos_a, mean_a, pos_b, mean_b, out_path, title):
        # Restrict to positions BOTH variants reached (intersection), so the
        # overlay never has to plot a missing value for either curve.
        common = sorted(set(pos_a) & set(pos_b))
        if not common:
            return
        map_a, map_b = dict(zip(pos_a, mean_a)), dict(zip(pos_b, mean_b))
        plots.plot_refusal_curves([map_a[p] for p in common], [map_b[p] for p in common], common,
                                   out_path, variant_a, variant_b, title=title)

    if mode in ("exact", "both") and any(sparse_curves["a"]):
        pos_a, mean_a = per_position_mean_sparse(sparse_curves["a"])
        pos_b, mean_b = per_position_mean_sparse(sparse_curves["b"])
        _plot_paired_curves(pos_a, mean_a, pos_b, mean_b, results_dir / f"{stem}_exact_curve.png",
                             f"Exact refusal probability: {variant_a} vs {variant_b}")

    if mode in ("token", "both") and any(dense_curves["a"]):
        pos_a, mean_a = per_position_mean_dense(dense_curves["a"])
        pos_b, mean_b = per_position_mean_dense(dense_curves["b"])
        _plot_paired_curves(pos_a, mean_a, pos_b, mean_b, results_dir / f"{stem}_token_curve.png",
                             f"Token-level (first-token) refusal probability: {variant_a} vs {variant_b}")

    if summary["a"]["n"] and summary["b"]["n"]:
        maxes_a = [p["stats_a"]["max"] for p in per_prompt if p["stats_a"] and p["stats_a"]["max"] is not None]
        aucs_a = [p["stats_a"]["auc"] for p in per_prompt if p["stats_a"] and p["stats_a"]["auc"] is not None]
        maxes_b = [p["stats_b"]["max"] for p in per_prompt if p["stats_b"] and p["stats_b"]["max"] is not None]
        aucs_b = [p["stats_b"]["auc"] for p in per_prompt if p["stats_b"] and p["stats_b"]["auc"] is not None]
        headline_stats_a = {
            "mean": summary["a"]["mean"],
            "max": max(maxes_a, default=0.0),
            "auc": summarize(aucs_a)["mean"],
        }
        headline_stats_b = {
            "mean": summary["b"]["mean"],
            "max": max(maxes_b, default=0.0),
            "auc": summarize(aucs_b)["mean"],
        }
        plots.plot_refusal_stats_bar(headline_stats_a, headline_stats_b, results_dir / f"{stem}_stats_bar.png",
                                      variant_a, variant_b)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Method 9: refusal-probability curves (teacher-forced) between two variants."
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--variant_a", required=True)
    parser.add_argument("--variant_b", required=True)
    parser.add_argument("--categories", nargs="+", default=["harmful", "harmless"])
    parser.add_argument("--jailbreak_file", default=None)
    parser.add_argument("--n_prompts", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--phrases", nargs="+", default=None,
                         help="Custom refusal phrase list (default: refusal.DEFAULT_REFUSAL_PHRASES)")
    parser.add_argument("--stride", type=int, default=4,
                         help="Evaluate exact multi-token refusal probability every `stride` generated positions "
                              "(1 = every position, most accurate but most forward passes). Ignored in mode='token'.")
    parser.add_argument("--mode", choices=["token", "exact", "both"], default="both",
                         help="'exact' = true multi-token joint probability (extra forward passes); "
                              "'token' = free first-token-only baseline; 'both' computes and plots each.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--alpha_override", type=float, default=None)
    args = parser.parse_args()
    run(args.model, args.variant_a, args.variant_b, args.categories, args.n_prompts, args.max_new_tokens,
        args.phrases, args.stride, args.mode, args.seed, args.jailbreak_file, args.label, args.output_dir,
        args.alpha_override)


if __name__ == "__main__":
    main()
