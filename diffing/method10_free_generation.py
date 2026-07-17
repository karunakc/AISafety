"""
Method 10: FREE (unconstrained) generation comparison between two variants,
evaluated over a dataset of prompts -- the counterpart to method8_jsd.py's
teacher-forced comparison and method9_refusal_probability.py's teacher-forced
refusal analysis.

Both variants generate independently from the same prompt with IDENTICAL
decoding parameters. For each prompt this computes:
  - divergence position: the first token index at which the two free
    generations differ (== common prefix length) -- see
    free_generation.divergence_position
  - JSD between the two models' per-step logits, up to min(len_a, len_b).
    IMPORTANT: only positions <= the divergence position are a fair "same
    context, different model" comparison, matching method8's teacher-forced
    JSD. Past the divergence position each model is conditioned on ITS OWN
    prior tokens, so the two logit rows being compared came from different
    contexts -- the JSD number is still computed and plotted (it's not
    meaningless: "how differently would each model score its own trajectory
    from here" is a fair question), but it is no longer directly comparable
    to method8's number and every plot here marks the divergence position so
    this distinction is never silently lost.
  - top-k candidate overlap (diffing/free_generation.py::topk_overlap):
    cheap and complements JSD -- JSD says the distributions differ in mass,
    top-k overlap says whether they differ in WHICH tokens are even
    considered plausible.
  - refusal-probability curves (reusing diffing/refusal.py exactly as
    method9 does, just applied to each model's own free-generation context
    instead of a shared teacher-forced one).

Across the dataset this aggregates the distribution of divergence positions
(so you can see, e.g., "half of harmful prompts diverge within the first 3
tokens") and picks out a handful of prompts with the earliest divergence as
qualitative examples.

Usage:
    python diffing/method10_free_generation.py --model Qwen/Qwen3.5-4B \\
        --variant_a base --variant_b M2 --categories harmful --n_prompts 50

    # Sampling instead of greedy (same seed/params reused for both variants
    # so "identical decoding parameters" holds):
    python diffing/method10_free_generation.py --model Qwen/Qwen3.5-4B \\
        --variant_a base --variant_b M2 --do_sample --temperature 0.7 --top_p 0.9
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "evaluations"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_common import load_variant, model_slug, remove_hooks  # noqa: E402

import plots  # noqa: E402
from curve_utils import per_position_mean_dense, per_position_mean_sparse  # noqa: E402
from free_generation import divergence_position, free_generate, topk_overlap  # noqa: E402
from jsd import METRICS  # noqa: E402
from method8_jsd import load_prompts, summarize  # noqa: E402
from refusal import (  # noqa: E402
    DEFAULT_REFUSAL_PHRASES,
    refusal_curve_stats,
    refusal_curves_for_model,
    tokenize_phrases,
)
from teacher_forcing import build_chat_prefix  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _full_ids(prefix_ids, gen_token_ids, device):
    """Reconstructs the [1, seq_len] (prefix + this model's OWN generated
    continuation) sequence needed by refusal.exact_refusal_curve's
    teacher-forcing, on `device`. Unlike method9 (where full_ids is shared
    between both models), each free-generation model gets its own."""
    return torch.cat([prefix_ids[0].cpu(), gen_token_ids]).unsqueeze(0).to(device)


@torch.no_grad()
def compare_free_generations(model_a, model_b, tokenizer, prompt, max_new_tokens=64, do_sample=False,
                              temperature=1.0, top_p=1.0, seed=42, topk=10, phrase_ids=None,
                              refusal_stride=4, refusal_mode="both"):
    """Runs the full Part 2 comparison for ONE prompt: independent generation
    from both models with identical decoding parameters, then divergence
    position, JSD, top-k overlap (over the shared min(len_a, len_b) span),
    and per-model refusal-probability curves (over each model's own full
    generation). Returns a dict of raw per-position arrays/scalars; no
    plotting or dataset-level aggregation happens here (see `run` below).
    """
    prefix_ids = build_chat_prefix(tokenizer, prompt, model_a.device)
    gen_a = free_generate(model_a, tokenizer, prefix_ids, max_new_tokens, do_sample, temperature, top_p, seed)
    prefix_ids_b = prefix_ids.to(model_b.device)
    gen_b = free_generate(model_b, tokenizer, prefix_ids_b, max_new_tokens, do_sample, temperature, top_p, seed)

    common_prefix_len, diverged = divergence_position(gen_a["token_ids"], gen_b["token_ids"])
    min_len = min(gen_a["logits"].shape[0], gen_b["logits"].shape[0])

    jsd_curve = topk_curve = None
    if min_len > 0:
        logits_a, logits_b = gen_a["logits"][:min_len], gen_b["logits"][:min_len]
        jsd_curve = METRICS["jsd"](logits_a, logits_b, reduction="none")
        topk_curve = topk_overlap(logits_a, logits_b, k=topk)

    refusal_a = refusal_b = None
    if phrase_ids is not None:
        start = prefix_ids.shape[1] - 1
        gen_len_a, gen_len_b = gen_a["token_ids"].shape[0], gen_b["token_ids"].shape[0]
        refusal_a = refusal_curves_for_model(
            model_a, gen_a["logits"], _full_ids(prefix_ids, gen_a["token_ids"], model_a.device),
            start, gen_len_a, phrase_ids, refusal_stride, refusal_mode,
        )
        refusal_b = refusal_curves_for_model(
            model_b, gen_b["logits"], _full_ids(prefix_ids_b, gen_b["token_ids"], model_b.device),
            start, gen_len_b, phrase_ids, refusal_stride, refusal_mode,
        )

    return {
        "text_a": gen_a["text"], "text_b": gen_b["text"],
        "token_ids_a": gen_a["token_ids"], "token_ids_b": gen_b["token_ids"],
        "common_prefix_len": common_prefix_len, "diverged": diverged,
        "jsd_curve": jsd_curve, "topk_curve": topk_curve,
        "refusal_a": refusal_a, "refusal_b": refusal_b,
    }


def run(model, variant_a, variant_b, categories=None, n_prompts=50, max_new_tokens=64,
        do_sample=False, temperature=1.0, top_p=1.0, topk=10, phrases=None, refusal_stride=4,
        refusal_mode="both", seed=42, jailbreak_file=None, label=None, output_dir=None,
        alpha_override=None, n_examples=4):
    categories = categories or ["harmful", "harmless"]
    phrases = list(phrases) if phrases else list(DEFAULT_REFUSAL_PHRASES)
    prompts = load_prompts(categories, n_prompts, seed, jailbreak_file)

    model_a, tokenizer, handles_a = load_variant(model, variant_a, alpha_override=alpha_override)
    model_b, _, handles_b = load_variant(model, variant_b, alpha_override=alpha_override)
    phrase_ids = tokenize_phrases(tokenizer, phrases)
    headline_key = "exact_curve" if refusal_mode in ("exact", "both") else "token_curve"

    per_prompt = []
    jsd_curves, topk_curves = [], []
    dense_refusal = {"a": [], "b": []}
    sparse_refusal = {"a": [], "b": []}
    divergence_positions = []

    try:
        desc = f"free_gen({variant_a} || {variant_b})"
        for prompt, category in tqdm(prompts, desc=desc):
            cmp = compare_free_generations(
                model_a, model_b, tokenizer, prompt, max_new_tokens, do_sample, temperature, top_p,
                seed, topk, phrase_ids, refusal_stride, refusal_mode,
            )
            if cmp["token_ids_a"].numel() == 0 and cmp["token_ids_b"].numel() == 0:
                continue

            divergence_positions.append(cmp["common_prefix_len"])
            entry = {
                "prompt": prompt, "category": category,
                "text_a": cmp["text_a"], "text_b": cmp["text_b"],
                "common_prefix_len": cmp["common_prefix_len"], "diverged": cmp["diverged"],
                "gen_len_a": cmp["token_ids_a"].shape[0], "gen_len_b": cmp["token_ids_b"].shape[0],
            }
            if cmp["jsd_curve"] is not None:
                entry["jsd_curve"] = cmp["jsd_curve"].tolist()
                entry["topk_curve"] = cmp["topk_curve"].tolist()
                jsd_curves.append(cmp["jsd_curve"])
                topk_curves.append(cmp["topk_curve"])

            for name in ("a", "b"):
                res = cmp[f"refusal_{name}"]
                if res is None:
                    continue
                if "token_curve" in res:
                    entry[f"refusal_token_curve_{name}"] = res["token_curve"].tolist()
                    dense_refusal[name].append(res["token_curve"])
                if "exact_curve" in res:
                    entry[f"refusal_exact_curve_{name}"] = res["exact_curve"].tolist()
                    entry[f"refusal_exact_positions_{name}"] = res["exact_positions"]
                    sparse_refusal[name].append((res["exact_curve"], res["exact_positions"]))
                headline_curve = res.get(headline_key)
                headline_positions = res.get("exact_positions") if headline_key == "exact_curve" else None
                entry[f"refusal_stats_{name}"] = (
                    refusal_curve_stats(headline_curve, headline_positions) if headline_curve is not None else None
                )
            per_prompt.append(entry)
    finally:
        remove_hooks(handles_a)
        remove_hooks(handles_b)

    results_dir = Path(output_dir) if output_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = label or f"{model_slug(model)}__{variant_a}_vs_{variant_b}__free_gen"
    if alpha_override is not None:
        stem += f"__alpha{alpha_override}"

    divergence_summary = summarize(divergence_positions)
    refusal_summary = {
        name: summarize([
            p[f"refusal_stats_{name}"]["mean"] for p in per_prompt
            if p.get(f"refusal_stats_{name}") and p[f"refusal_stats_{name}"]["mean"] is not None
        ])
        for name in ("a", "b")
    }

    # Representative examples: prompts where the two free generations
    # diverged earliest (smallest common_prefix_len among prompts that
    # actually diverged, rather than one just running out of tokens first).
    diverged_prompts = [p for p in per_prompt if p["diverged"]]
    examples = sorted(diverged_prompts, key=lambda p: p["common_prefix_len"])[:n_examples]

    result = {
        "method": "free_generation", "model": model, "variant_a": variant_a, "variant_b": variant_b,
        "phrases": phrases, "refusal_mode": refusal_mode, "refusal_stride": refusal_stride,
        "do_sample": do_sample, "temperature": temperature, "top_p": top_p, "topk": topk,
        "alpha_override": alpha_override, "n_prompts": len(per_prompt),
        "divergence_summary": divergence_summary, "refusal_summary": refusal_summary,
        "per_prompt": per_prompt,
    }

    json_path = results_dir / f"{stem}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to {json_path}")
    print(f"divergence_summary: {divergence_summary}")
    print(f"refusal_summary: {refusal_summary}")

    plots.plot_divergence_histogram(divergence_positions, results_dir / f"{stem}_divergence_hist.png",
                                     variant_a, variant_b)

    if jsd_curves:
        positions, mean_jsd = per_position_mean_dense(jsd_curves)
        median_divergence = int(divergence_summary["median"]) if divergence_summary["median"] is not None else None
        plots.plot_jsd_with_divergence_marker(mean_jsd, results_dir / f"{stem}_jsd.png", variant_a, variant_b,
                                               divergence_position=median_divergence)
        _, mean_topk = per_position_mean_dense(topk_curves)
        plots.plot_topk_overlap(mean_topk, results_dir / f"{stem}_topk_overlap.png", variant_a, variant_b, topk,
                                 divergence_position=median_divergence)

    def _plot_paired_refusal(pos_a, mean_a, pos_b, mean_b, out_path, title):
        common = sorted(set(pos_a) & set(pos_b))
        if not common:
            return
        map_a, map_b = dict(zip(pos_a, mean_a)), dict(zip(pos_b, mean_b))
        plots.plot_refusal_curves([map_a[p] for p in common], [map_b[p] for p in common], common,
                                   out_path, variant_a, variant_b, title=title)

    if refusal_mode in ("exact", "both") and any(sparse_refusal["a"]):
        pos_a, mean_a = per_position_mean_sparse(sparse_refusal["a"])
        pos_b, mean_b = per_position_mean_sparse(sparse_refusal["b"])
        _plot_paired_refusal(pos_a, mean_a, pos_b, mean_b, results_dir / f"{stem}_refusal_exact.png",
                              f"Free-generation exact refusal probability: {variant_a} vs {variant_b}")

    if refusal_mode in ("token", "both") and any(dense_refusal["a"]):
        pos_a, mean_a = per_position_mean_dense(dense_refusal["a"])
        pos_b, mean_b = per_position_mean_dense(dense_refusal["b"])
        _plot_paired_refusal(pos_a, mean_a, pos_b, mean_b, results_dir / f"{stem}_refusal_token.png",
                              f"Free-generation token-level refusal probability: {variant_a} vs {variant_b}")

    if examples:
        plots.plot_representative_examples(
            [{"prompt": e["prompt"], "text_a": e["text_a"], "text_b": e["text_b"],
              "diverged_at": e["common_prefix_len"]} for e in examples],
            results_dir / f"{stem}_examples.png", variant_a, variant_b,
        )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Method 10: free-generation comparison (divergence, JSD, top-k overlap, refusal probability)."
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--variant_a", required=True)
    parser.add_argument("--variant_b", required=True)
    parser.add_argument("--categories", nargs="+", default=["harmful", "harmless"])
    parser.add_argument("--jailbreak_file", default=None)
    parser.add_argument("--n_prompts", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--do_sample", action="store_true", help="Sample instead of greedy decoding")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=10, help="k for top-k candidate overlap")
    parser.add_argument("--phrases", nargs="+", default=None,
                         help="Custom refusal phrase list (default: refusal.DEFAULT_REFUSAL_PHRASES)")
    parser.add_argument("--refusal_stride", type=int, default=4,
                         help="Evaluate exact multi-token refusal probability every N generated positions")
    parser.add_argument("--refusal_mode", choices=["token", "exact", "both"], default="both")
    parser.add_argument("--n_examples", type=int, default=4,
                         help="Number of earliest-diverging prompts to render as qualitative examples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--alpha_override", type=float, default=None)
    args = parser.parse_args()
    run(args.model, args.variant_a, args.variant_b, args.categories, args.n_prompts, args.max_new_tokens,
        args.do_sample, args.temperature, args.top_p, args.topk, args.phrases, args.refusal_stride,
        args.refusal_mode, args.seed, args.jailbreak_file, args.label, args.output_dir, args.alpha_override,
        args.n_examples)


if __name__ == "__main__":
    main()
