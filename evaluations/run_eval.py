"""
Evaluate a given (model, variant) pair across the capability / safety /
emotion / OOD benchmark suite from the evaluation plan.

Usage:
    python evaluations/run_eval.py --model Qwen/Qwen2.5-7B-Instruct --variant base
    python evaluations/run_eval.py --model Qwen/Qwen2.5-7B-Instruct --variant M2.1 --categories safety emotion
"""

import argparse
import json
import traceback
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


def run(model, variant, categories=None, n_prompts=100, limit=None, output=None, direction_source=None):
    """Core logic, callable directly (e.g. from modal/modal_app.py) without going through argparse.

    `direction_source`, if given, steers `model` using a different model's
    saved M2.x/M3.x direction.pt instead of `model`'s own (see
    eval_common.load_variant)."""
    categories = categories or list(CATEGORY_RUNNERS)

    output_path = Path(output) if output else RESULTS_DIR / f"{model_slug(model)}_{variant}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {"model": model, "variant": variant, "direction_source": direction_source}

    def _save():
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Wrote results to {output_path}")

    causal_model, tokenizer, handles = load_variant(model, variant, direction_source=direction_source)
    try:
        for category in categories:
            print(f"=== Running {category} benchmarks for {model} [{variant}] ===")
            try:
                results[category] = CATEGORY_RUNNERS[category](causal_model, tokenizer, n_prompts, limit)
            except Exception as e:
                print(f"ERROR: {category} benchmark failed, continuing with remaining categories: {e}")
                traceback.print_exc()
                results[category] = {"error": str(e)}
            finally:
                # Save after every category so a later failure never loses
                # results that already succeeded.
                _save()
    finally:
        remove_hooks(handles)

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
    parser.add_argument("--output", default=None)
    parser.add_argument("--direction_source", default=None,
                         help="Steer `model` using a different model's saved direction.pt "
                              "instead of model's own (e.g. the base model's M2.1 direction)")
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
