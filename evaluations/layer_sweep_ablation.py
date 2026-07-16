"""
Layer sweep for M2.2-style directional ablation.

Uses refusal_misaligned_simple.py's own method (AdvBench/Alpaca prompts,
capture_layer_activation + compute_direction -- single fixed layer, no
judge model, no filtering) to probe a refusal direction AT EACH CANDIDATE
LAYER in turn, then applies directional ablation with THAT direction at
every layer (same mechanism as M2.2/register_ablation_steering_hooks) and
measures HarmBench attack success rate -- to see which source layer's
direction is most effective when used for full-network ablation.

Usage:
    python evaluations/layer_sweep_ablation.py --model Qwen/Qwen3.5-4B
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from common import (  # noqa: E402
    capture_layer_activation,
    compute_direction,
    get_decoder_layers,
    get_device,
    load_model_and_tokenizer,
    model_slug,
    register_ablation_steering_hooks,
    remove_hooks,
)
from refusal_misaligned_simple import get_harmful_prompts, get_harmless_prompts  # noqa: E402
from safety import run_safety_benchmark

RESULTS_DIR = PROJECT_ROOT / "results"


def run(model, n_prompts=64, eval_n_prompts=10, eval_max_new_tokens=512, layers=None,
        enable_thinking=False, output=None):
    device = get_device()
    causal_model, tokenizer = load_model_and_tokenizer(model, device=device)
    n_layers = len(get_decoder_layers(causal_model))
    sweep_layers = layers if layers is not None else list(range(n_layers))

    harmful_prompts = get_harmful_prompts(n_prompts)
    harmless_prompts = get_harmless_prompts(n_prompts)

    results = {}
    for l in sweep_layers:
        harmful_acts = [
            capture_layer_activation(causal_model, tokenizer, p, l, enable_thinking=enable_thinking)
            for p in tqdm(harmful_prompts, desc=f"layer {l}: probing harmful")
        ]
        harmless_acts = [
            capture_layer_activation(causal_model, tokenizer, p, l, enable_thinking=enable_thinking)
            for p in tqdm(harmless_prompts, desc=f"layer {l}: probing harmless")
        ]
        direction = compute_direction(pos_activations=harmful_acts, neg_activations=harmless_acts)

        handles = register_ablation_steering_hooks(causal_model, direction, list(range(n_layers)))
        try:
            r = run_safety_benchmark(
                causal_model, tokenizer, "harmbench", n_prompts=eval_n_prompts,
                max_new_tokens=eval_max_new_tokens, enable_thinking=enable_thinking,
            )
        finally:
            remove_hooks(handles)

        results[l] = {
            "attack_success_rate": r["attack_success_rate"],
            "mean_strongreject_score": r["mean_strongreject_score"],
        }
        print(f"layer {l}: ASR={r['attack_success_rate']:.3f}, mean_score={r['mean_strongreject_score']:.3f}")

    out_path = Path(output) if output else RESULTS_DIR / f"{model_slug(model)}_ablation_layer_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "model": model,
                "eval_n_prompts": eval_n_prompts,
                "eval_max_new_tokens": eval_max_new_tokens,
                "results": {str(k): v for k, v in results.items()},
            },
            f, indent=2,
        )
    print(f"Wrote sweep results to {out_path}")

    best_layer = min(results, key=lambda l: results[l]["mean_strongreject_score"])
    print(f"Most effective layer (lowest mean StrongReject score): {best_layer} -> {results[best_layer]}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Layer sweep for directional ablation, HarmBench-scored.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n_prompts", type=int, default=64,
                         help="Harmful/harmless prompts used to probe each candidate layer's direction")
    parser.add_argument("--eval_n_prompts", type=int, default=10, help="HarmBench prompts per layer for the sweep")
    parser.add_argument("--eval_max_new_tokens", type=int, default=512)
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                         help="Specific layers to sweep (default: every layer)")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
