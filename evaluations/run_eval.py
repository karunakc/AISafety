"""
CLI entrypoint: evaluate a given (model, variant) pair across the
capability / safety / emotion / OOD benchmark suite from the evaluation plan.

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
    "capability": lambda model, tokenizer, args: run_capability_benchmarks(model, tokenizer, limit=args.limit),
    "safety": lambda model, tokenizer, args: run_safety_benchmarks(model, tokenizer, n_prompts=args.n_prompts),
    "emotion": lambda model, tokenizer, args: run_emotion_benchmarks(model, tokenizer, n_prompts=args.n_prompts),
    "ood": lambda model, tokenizer, args: run_ood_benchmark(model, tokenizer, device=get_device()),
}


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
    args = parser.parse_args()

    model, tokenizer, handles = load_variant(args.model, args.variant)
    try:
        results = {"model": args.model, "variant": args.variant}
        for category in args.categories:
            print(f"=== Running {category} benchmarks for {args.model} [{args.variant}] ===")
            results[category] = CATEGORY_RUNNERS[category](model, tokenizer, args)
    finally:
        remove_hooks(handles)

    output_path = Path(args.output) if args.output else RESULTS_DIR / f"{model_slug(args.model)}_{args.variant}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
