"""
Evaluate a given (model, variant) pair across the capability / safety / OOD
benchmark suite from the evaluation plan.

Usage:
    python evaluations/run_eval.py --model Qwen/Qwen2.5-7B-Instruct --variant base
    python evaluations/run_eval.py --model Qwen/Qwen2.5-7B-Instruct --variant M2 --categories safety ood
"""

import argparse
import json
import traceback
from pathlib import Path

from capability import CAPABILITY_TASKS, run_capability_benchmarks
from eval_common import PROJECT_ROOT, VARIANTS, get_device, load_variant, model_slug, remove_hooks
from ood import run_ood_benchmark
from safety import run_safety_benchmarks

RESULTS_DIR = PROJECT_ROOT / "results"

CATEGORY_RUNNERS = {
    "capability": lambda model, tokenizer, cfg: run_capability_benchmarks(
        model, tokenizer, tasks=cfg["capability_tasks"], limit=cfg["limit"],
        mmlu_pro_total_limit=cfg["mmlu_pro_total_limit"]),
    "safety": lambda model, tokenizer, cfg: run_safety_benchmarks(
        model, tokenizer, n_prompts=cfg["n_prompts"], max_new_tokens=cfg["max_new_tokens"],
        n_generations=cfg["n_generations"], success_threshold=cfg["success_threshold"],
        enable_thinking=cfg["enable_thinking"]),
    "ood": lambda model, tokenizer, cfg: run_ood_benchmark(model, tokenizer, device=get_device(),
                                                            enable_thinking=cfg["enable_thinking"]),
}


def run(model, variant, categories=None, capability_tasks=None, n_prompts=100, limit=None,
        mmlu_pro_total_limit=None, max_new_tokens=128, n_generations=1, success_threshold=None,
        enable_thinking=False, alpha_override=None, output=None):
    """Core logic, callable directly (e.g. from modal/modal_app.py) without going through argparse."""
    categories = categories or list(CATEGORY_RUNNERS)
    cfg = dict(capability_tasks=capability_tasks, n_prompts=n_prompts, limit=limit,
               mmlu_pro_total_limit=mmlu_pro_total_limit, max_new_tokens=max_new_tokens,
               n_generations=n_generations, success_threshold=success_threshold,
               enable_thinking=enable_thinking)

    # Thinking-off is the historical default and keeps the existing filename
    # convention (no suffix); thinking-on gets an explicit suffix so it can't
    # collide with or silently overwrite a thinking-off result for the same variant.
    default_stem = f"{model_slug(model)}_{variant}" + ("_thinking" if enable_thinking else "")
    output_path = Path(output) if output else RESULTS_DIR / f"{default_stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {"model": model, "variant": variant}

    def _save():
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Wrote results to {output_path}")

    causal_model, tokenizer, handles = load_variant(model, variant, alpha_override=alpha_override)
    try:
        for category in categories:
            print(f"=== Running {category} benchmarks for {model} [{variant}] ===")
            try:
                results[category] = CATEGORY_RUNNERS[category](causal_model, tokenizer, cfg)
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
        description="Evaluate a (model, variant) pair over the capability/safety/OOD benchmark suite."
    )
    parser.add_argument("--model", required=True, help="Base model name, e.g. Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--categories", nargs="+", default=list(CATEGORY_RUNNERS), choices=list(CATEGORY_RUNNERS))
    parser.add_argument("--capability_tasks", nargs="+", default=None, choices=CAPABILITY_TASKS,
                         help="Which capability tasks to run (default: all of %(choices)s)")
    parser.add_argument("--n_prompts", type=int, default=100, help="Prompts per safety benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Optional example cap per capability SUBTASK (quick runs) -- "
                         "mmlu_pro/bbh_cot_fewshot are groups of many subtasks, so this caps each individually, "
                         "not the group total (see --mmlu_pro_total_limit)")
    parser.add_argument("--mmlu_pro_total_limit", type=int, default=None,
                         help="Cap mmlu_pro's TOTAL example count summed across all 14 subject subtasks "
                              "(each subtask trimmed proportionally to its own size). Overrides --limit for "
                              "mmlu_pro specifically; --limit still applies to gsm8k/bbh_cot_fewshot.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens per safety generation")
    parser.add_argument("--n_generations", type=int, default=1, help="Sampled generations per safety prompt")
    parser.add_argument("--success_threshold", type=int, default=None,
                         help="A safety prompt counts as an attack success if MORE than this many of its "
                              "n_generations are individually judged harmful (default: majority, n_generations // 2)")
    parser.add_argument("--alpha_override", type=float, default=None,
                         help="Override M1_risky+M2's steering-towards-refusal coefficient magnitude "
                              "(default: abs(M2's own saved coef)). Ignored by every other variant.")
    parser.add_argument("--enable_thinking", action="store_true",
                         help="Enable thinking mode in the chat template for safety/ood generations "
                              "(default off). Adds a '_thinking' suffix to the default output filename.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
