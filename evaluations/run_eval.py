"""
Evaluate a given (model, variant) pair across the capability / safety /
emotion / OOD benchmark suite from the evaluation plan.

Usage:
    python evaluations/run_eval.py --model Qwen/Qwen2.5-7B-Instruct --variant base
    python evaluations/run_eval.py --model Qwen/Qwen2.5-7B-Instruct --variant M2.1 --categories safety emotion
"""

import argparse
import json
from pathlib import Path

from capability import run_capability_benchmarks
from eval_common import PROJECT_ROOT, VARIANTS, get_device, load_variant, model_slug, remove_hooks
from emotion import run_emotion_benchmarks
from ood import run_ood_benchmark
from safety import run_safety_benchmarks

RESULTS_DIR = PROJECT_ROOT / "results"

CATEGORY_RUNNERS = {
    "capability": lambda model, tokenizer, n_prompts, limit: run_capability_benchmarks(model, tokenizer, limit=limit),
    "safety": lambda model, tokenizer, n_prompts, limit: run_safety_benchmarks(model, tokenizer, n_prompts=n_prompts),
    "emotion": lambda model, tokenizer, n_prompts, limit: run_emotion_benchmarks(model, tokenizer, n_prompts=n_prompts),
    "ood": lambda model, tokenizer, n_prompts, limit: run_ood_benchmark(model, tokenizer, device=get_device()),
}


def run(model, variant, categories=None, n_prompts=100, limit=None, output=None):
    """Core logic, callable directly (e.g. from modal/modal_app.py) without going through argparse."""
    categories = categories or list(CATEGORY_RUNNERS)

    causal_model, tokenizer, handles = load_variant(model, variant, theta_deg=0.0 if variant in ["M3.1", "M3.2"] else None, coef=None if variant in ["M2.1", "M3.1"] else None)
    try:
        results = {"model": model, "variant": variant}
        for category in categories:
            print(f"=== Running {category} benchmarks for {model} [{variant}] ===")
            results[category] = CATEGORY_RUNNERS[category](causal_model, tokenizer, n_prompts, limit)
    finally:
        remove_hooks(handles)

    output_path = Path(output) if output else RESULTS_DIR / f"{model_slug(model)}_{variant}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Wrote results to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a (model, variant) pair over the capability/safety/emotion/OOD benchmark suite."
    )
    parser.add_argument("--model", required=True, help="Base model name, e.g. Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--categories", nargs="+", default=list(CATEGORY_RUNNERS), choices=list(CATEGORY_RUNNERS))
    parser.add_argument("--n_prompts", type=int, default=100, help="Prompts per safety/emotion benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Optional example cap per capability task (quick runs)")
    parser.add_argument("--theta_deg", type=float, default=0.0, help="Optional override for angular steering angle (M3.1/M3.2 only)")
    parser.add_argument("--coef", type=float, default=None, help="Optional override for additive steering coefficient (M2.1/M3.1 only)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
