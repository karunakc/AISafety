"""
Re-run just the judge-scored alpha search (Step 7 of refusal_misaligned.py)
against an already-saved M2.1 direction, without redoing refusal-token
detection, prompt filtering/splitting, activation extraction, probing, or
direction selection (Steps 1-6).

Only alpha is searched here -- the direction and layer are read back exactly
as already selected in a prior full run, not recomputed.

Requires a prior full run of refusal_misaligned.py for --model, since it
reads back:
  - data/refusal/harmful_val.json                                  (Step 2's split)
  - models/<slug>/M2.1_steer_against_refusal_additive/direction.pt  (Step 8's save:
    direction = best_raw_direction, layers = [best_layer])

Re-saves M2.1's direction.pt with the new judge-selected alpha. The direction
and layers written back are identical to what was read in -- torch.save
writes the whole dict as one file, so there's no way to patch just the coef
field without rewriting the file, but nothing about the direction itself is
recomputed. Does not touch M2.2 (angular), since that artifact doesn't
depend on alpha at all.

Usage:
    python scripts/rerun_alpha_judge_search.py --model Qwen/Qwen3-4B-Instruct-2507 \\
        --judge_model Qwen/Qwen2.5-0.5B-Instruct \\
        --alpha_grid -0.5 -1 -2 -5 -10
"""

import argparse
import json

import torch

from alpha_judge_search import alpha_judge_search, plot_alpha_judge_search
from common import (
    DATA_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    get_device,
    load_direction,
    load_model_and_tokenizer,
    model_slug,
    save_direction,
)

PLOTS_DIR = PROJECT_ROOT / "plots"
SPLITS_DIR = DATA_DIR / "refusal"


def run(
    model,
    judge_model,
    alpha_grid=None,
    coherence_threshold=6.0,
    alpha_search_n_prompts=16,
    alpha_search_max_new_tokens=128,
    enable_thinking=False,
):
    if alpha_grid is None:
        alpha_grid = [-0.5, -1.0, -1.5, -2.0, -5.0]

    print(f"=== Re-run judge-scored alpha search for {model} ===")
    print(f"alpha_grid={alpha_grid}, coherence_threshold={coherence_threshold}, "
          f"n_prompts={alpha_search_n_prompts}, max_new_tokens={alpha_search_max_new_tokens}, "
          f"enable_thinking={enable_thinking}")

    device = get_device()
    print(f"Device: {device}")
    slug = model_slug(model)

    harmful_val_path = SPLITS_DIR / "harmful_val.json"
    if not harmful_val_path.exists():
        raise FileNotFoundError(
            f"No saved split at {harmful_val_path}. Run scripts/refusal_misaligned.py --model {model} "
            "first (Steps 1-6) to produce it."
        )
    harmful_val = json.load(open(harmful_val_path))
    print(f"Loaded {len(harmful_val)} harmful_val prompts from {harmful_val_path}")

    additive_path = MODELS_DIR / slug / "M2.1_steer_against_refusal_additive" / "direction.pt"
    if not additive_path.exists():
        raise FileNotFoundError(
            f"No saved M2.1 direction at {additive_path}. Run scripts/refusal_misaligned.py --model {model} "
            "first to produce a direction/layer to re-tune alpha against."
        )
    saved = load_direction(additive_path)
    best_raw_direction = saved["direction"]
    best_layer = saved["layers"][0]
    print(f"Loaded existing M2.1 direction from {additive_path}: layer={best_layer}, previous alpha={saved['coef']}")

    print(f"Loading model: {model}")
    model_obj, tokenizer = load_model_and_tokenizer(model, device=device)
    print(f"Model loaded on {device}")

    print(f"Loading judge model: {judge_model}")
    judge_model_obj, judge_tokenizer = load_model_and_tokenizer(judge_model, device=device)
    print(f"Judge model loaded on {device}")
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
        print("Unloaded judge model")

    plot_alpha_judge_search(alpha_judge_results, best_alpha, PLOTS_DIR)
    if best_alpha is None:
        print(f"WARNING: No alpha cleared coherence_threshold={coherence_threshold}; "
              f"falling back to lowest-refusal alpha regardless of coherence.")
        best_alpha = min(alpha_judge_results, key=lambda a: alpha_judge_results[a]["refusal"])
    print(f"Best alpha: {best_alpha} "
          f"(refusal={alpha_judge_results[best_alpha]['refusal']:.2f}, "
          f"coherence={alpha_judge_results[best_alpha]['coherence']:.2f})")

    save_direction(best_raw_direction, best_alpha, "additive", saved["layers"], additive_path)
    print(f"Re-saved M2.1 additive (alpha={best_alpha}, layers={saved['layers']}) to {additive_path}")
    return additive_path


def main():
    parser = argparse.ArgumentParser(
        description="Re-run judge-scored alpha search against an already-saved M2.1 direction."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--judge_model", required=True)
    parser.add_argument("--alpha_grid", type=float, nargs="+",
                        default=[-0.5, -1.0, -1.5, -2.0, -5.0])
    parser.add_argument("--coherence_threshold", type=float, default=6.0)
    parser.add_argument("--alpha_search_n_prompts", type=int, default=16)
    parser.add_argument("--alpha_search_max_new_tokens", type=int, default=128)
    parser.add_argument("--enable_thinking", action="store_true")
    args = parser.parse_args()

    run(
        args.model,
        args.judge_model,
        alpha_grid=args.alpha_grid,
        coherence_threshold=args.coherence_threshold,
        alpha_search_n_prompts=args.alpha_search_n_prompts,
        alpha_search_max_new_tokens=args.alpha_search_max_new_tokens,
        enable_thinking=args.enable_thinking,
    )


if __name__ == "__main__":
    main()
