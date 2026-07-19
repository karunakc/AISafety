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


def _thinking_suffix(thinking):
    return "thinking" if thinking else "nothinking"

CATEGORY_RUNNERS = {
    "capability": lambda model, tokenizer, n_prompts, limit, mmlu_pro_limit, enable_thinking, n_responses: run_capability_benchmarks(
        model, tokenizer, limit=limit, mmlu_pro_limit=mmlu_pro_limit, enable_thinking=enable_thinking
    ),
    "safety": lambda model, tokenizer, n_prompts, limit, mmlu_pro_limit, enable_thinking, n_responses: run_safety_benchmarks(
        model, tokenizer, n_prompts=n_prompts, enable_thinking=enable_thinking, n_responses=n_responses
    ),
    "emotion": lambda model, tokenizer, n_prompts, limit, mmlu_pro_limit, enable_thinking, n_responses: run_emotion_benchmarks(
        model, tokenizer, n_prompts=n_prompts, enable_thinking=enable_thinking
    ),
    "ood": lambda model, tokenizer, n_prompts, limit, mmlu_pro_limit, enable_thinking, n_responses: run_ood_benchmark(
        model, tokenizer, device=get_device(), enable_thinking=enable_thinking
    ),
}


def _coef_suffix(coef_override):
    return f"_coef{coef_override}" if coef_override is not None else ""


def run(model, variant, categories=None, n_prompts=100, limit=None, mmlu_pro_limit=None, thinking=False,
        n_responses=10, output=None, direction_source=None, coef_override=None):
    """Core logic, callable directly (e.g. from modal/modal_app.py) without going through argparse.

    `direction_source`, if given, steers `model` using a different model's
    saved M2.x direction.pt instead of `model`'s own (see eval_common.load_variant).
    `coef_override`, if given, replaces the saved M2.1 coefficient (e.g. a
    positive value to induce refusal instead of the saved, judge-calibrated
    negative value that bypasses it) -- ignored for M2.2."""
    categories = categories or list(CATEGORY_RUNNERS)

    output_path = Path(output) if output else RESULTS_DIR / f"{model_slug(model)}_{variant}_{_thinking_suffix(thinking)}{_coef_suffix(coef_override)}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {"model": model, "variant": variant, "direction_source": direction_source,
               "coef_override": coef_override, "thinking_enabled": thinking}

    def _save():
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Wrote results to {output_path}")

    causal_model, tokenizer, handles = load_variant(model, variant, direction_source=direction_source,
                                                      coef_override=coef_override)
    try:
        for category in categories:
            print(f"=== Running {category} benchmarks for {model} [{variant}] ===")
            try:
                results[category] = CATEGORY_RUNNERS[category](causal_model, tokenizer, n_prompts, limit, mmlu_pro_limit, thinking, n_responses)
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


def run_category(model, variant, category, n_prompts=100, limit=None, mmlu_pro_limit=None, thinking=False,
                  n_responses=10, output=None, direction_source=None, coef_override=None):
    """Run a single benchmark category in isolation and write its own results
    file -- lets modal/modal_app.py fan a (model, variant) evaluation out
    across one GPU per category instead of running all four sequentially on
    one GPU."""
    causal_model, tokenizer, handles = load_variant(model, variant, direction_source=direction_source,
                                                      coef_override=coef_override)
    try:
        print(f"=== Running {category} benchmarks for {model} [{variant}] ===")
        result = {
            "model": model, "variant": variant, "direction_source": direction_source,
            "coef_override": coef_override, "thinking_enabled": thinking,
            category: CATEGORY_RUNNERS[category](causal_model, tokenizer, n_prompts, limit, mmlu_pro_limit, thinking, n_responses),
        }
    finally:
        remove_hooks(handles)

    output_path = Path(output) if output else RESULTS_DIR / f"{model_slug(model)}_{variant}_{category}_{_thinking_suffix(thinking)}{_coef_suffix(coef_override)}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Wrote results to {output_path}")
    return result


def merge_category_results(model, variant, categories=None, thinking=False, output=None, coef_override=None):
    """Combine the per-category JSON files written by run_category(...) into
    a single {model_slug}_{variant}_{thinking_suffix}.json matching run()'s output shape."""
    categories = categories or list(CATEGORY_RUNNERS)
    merged = {"model": model, "variant": variant}
    for category in categories:
        part_path = RESULTS_DIR / f"{model_slug(model)}_{variant}_{category}_{_thinking_suffix(thinking)}{_coef_suffix(coef_override)}.json"
        with open(part_path) as f:
            part = json.load(f)
        merged[category] = part[category]
        merged["thinking_enabled"] = part.get("thinking_enabled")

    output_path = Path(output) if output else RESULTS_DIR / f"{model_slug(model)}_{variant}_{_thinking_suffix(thinking)}{_coef_suffix(coef_override)}.json"
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
    parser.add_argument("--n_responses", type=int, default=10, help="Sampled responses per prompt for safety benchmarks, averaged before scoring")
    parser.add_argument("--limit", type=int, default=None, help="Optional example cap per capability task (quick runs)")
    parser.add_argument("--mmlu_pro_limit", type=int, default=None, help="Optional total example cap for mmlu_pro specifically (e.g. 1000 instead of the full ~12k)")
    parser.add_argument("--thinking", action="store_true", help="Enable model 'thinking' (Qwen3-style <think> traces) during generation; default is disabled")
    parser.add_argument("--output", default=None)
    parser.add_argument("--direction_source", default=None,
                         help="Steer `model` using a different model's saved direction.pt "
                              "instead of model's own (e.g. the base model's M2.1 direction)")
    parser.add_argument("--coef_override", type=float, default=None,
                         help="Replace the saved M2.1 coefficient (ignored for M2.2). The saved M2.1 "
                              "coefficient is calibrated to bypass refusal (negative); pass a positive "
                              "value here to instead induce refusal with the same direction/layers, "
                              "e.g. to reproduce the paper's positive-d 'refusal addition' intervention.")
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
