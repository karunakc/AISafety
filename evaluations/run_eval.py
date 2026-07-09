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
    "capability": lambda model, tokenizer, n_prompts, limit, mmlu_pro_limit: run_capability_benchmarks(
        model, tokenizer, limit=limit, mmlu_pro_limit=mmlu_pro_limit
    ),
    "safety": lambda model, tokenizer, n_prompts, limit, mmlu_pro_limit: run_safety_benchmarks(model, tokenizer, n_prompts=n_prompts),
    "emotion": lambda model, tokenizer, n_prompts, limit, mmlu_pro_limit: run_emotion_benchmarks(model, tokenizer, n_prompts=n_prompts),
    "ood": lambda model, tokenizer, n_prompts, limit, mmlu_pro_limit: run_ood_benchmark(model, tokenizer, device=get_device()),
}


def run(model, variant, categories=None, n_prompts=100, limit=None, mmlu_pro_limit=None, output=None):
    """Core logic, callable directly (e.g. from modal/modal_app.py) without going through argparse."""
    categories = categories or list(CATEGORY_RUNNERS)

    causal_model, tokenizer, handles = load_variant(model, variant)
    try:
        results = {"model": model, "variant": variant}
        for category in categories:
            print(f"=== Running {category} benchmarks for {model} [{variant}] ===")
            results[category] = CATEGORY_RUNNERS[category](causal_model, tokenizer, n_prompts, limit, mmlu_pro_limit)
    finally:
        remove_hooks(handles)

    output_path = Path(output) if output else RESULTS_DIR / f"{model_slug(model)}_{variant}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Wrote results to {output_path}")
    return results


def run_category(model, variant, category, n_prompts=100, limit=None, mmlu_pro_limit=None, output=None):
    """Run a single benchmark category in isolation and write its own results
    file -- lets modal/modal_app.py fan a (model, variant) evaluation out
    across one GPU per category instead of running all four sequentially on
    one GPU."""
    causal_model, tokenizer, handles = load_variant(model, variant)
    try:
        print(f"=== Running {category} benchmarks for {model} [{variant}] ===")
        result = {"model": model, "variant": variant, category: CATEGORY_RUNNERS[category](causal_model, tokenizer, n_prompts, limit, mmlu_pro_limit)}
    finally:
        remove_hooks(handles)

    output_path = Path(output) if output else RESULTS_DIR / f"{model_slug(model)}_{variant}_{category}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Wrote results to {output_path}")
    return result


def merge_category_results(model, variant, categories=None, output=None):
    """Combine the per-category JSON files written by run_category(...) into
    a single {model_slug}_{variant}.json matching run()'s output shape."""
    categories = categories or list(CATEGORY_RUNNERS)
    merged = {"model": model, "variant": variant}
    for category in categories:
        part_path = RESULTS_DIR / f"{model_slug(model)}_{variant}_{category}.json"
        with open(part_path) as f:
            merged[category] = json.load(f)[category]

    output_path = Path(output) if output else RESULTS_DIR / f"{model_slug(model)}_{variant}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"Wrote merged results to {output_path}")
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a (model, variant) pair over the capability/safety/emotion/OOD benchmark suite."
    )
    parser.add_argument("--model", required=True, help="Base model name, e.g. Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--categories", nargs="+", default=list(CATEGORY_RUNNERS), choices=list(CATEGORY_RUNNERS))
    parser.add_argument("--n_prompts", type=int, default=100, help="Prompts per safety/emotion benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Optional example cap per capability task (quick runs)")
    parser.add_argument("--mmlu_pro_limit", type=int, default=None, help="Optional total example cap for mmlu_pro specifically (e.g. 1000 instead of the full ~12k)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
