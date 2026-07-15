"""
Method 3, swept across every possible direction-source layer -- injecting
each direction AT the layer it was extracted from.

method3_induce.py picks ONE fixed direction (from a saved M2.1/M2.2 file, or
a single --layer override) and injects it at every injection layer in turn.
This script instead sweeps over EVERY possible direction-source layer L
(0..n_layers-1), but injects the layer-L direction ONLY at layer L itself --
not at every other injection layer too. Injecting a direction extracted at
layer L into some unrelated layer L' isn't a particularly meaningful
comparison; "how good is this layer's own direction, injected right where
it came from" is the natural per-layer question. This also makes the sweep
linear in layer count (n_direction_layers forward passes per prompt) instead
of quadratic (n_direction_layers * n_injection_layers), unlike an earlier
version of this script that swept the full injection-layer grid too.

The result is a single curve (induce score vs. layer L), tested vs. base
control -- the method3 analogue of method1_cosine_similarity.py's per-layer
curve, but for induce score instead of cosine similarity.

Usage:
    python diffing/method3_all_layers.py --model models/Qwen__Qwen3-4B/M2.3_ablation_baked
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import get_decoder_layers, get_device, load_model_and_tokenizer, model_slug  # noqa: E402
from refusal_misaligned import ACTIVATIONS_DIR, compute_directions, compute_induce_score, load_activations  # noqa: E402
from method3_induce import load_harmless_val, load_refusal_token_ids  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run(model, base_model="Qwen/Qwen3-4B", enable_thinking=False, output_dir=None, label=None):
    harmless_val = load_harmless_val(base_model)
    refusal_token_ids = load_refusal_token_ids(base_model)
    print(f"Loaded {len(harmless_val)} harmless_val prompts, {len(refusal_token_ids)} refusal tokens -- both from {base_model}")

    train_acts_dir = ACTIVATIONS_DIR / model_slug(base_model)
    if not (train_acts_dir / "harmful_train.pt").exists():
        raise FileNotFoundError(
            f"No cached train activations at {train_acts_dir}. Run scripts/refusal_misaligned.py --model {base_model} first."
        )
    harmful_train_acts, harmless_train_acts = load_activations(train_acts_dir, "train")
    _, raw_directions = compute_directions(harmful_train_acts, harmless_train_acts)  # RAW, not unit -- induce needs real magnitude
    n_direction_layers = raw_directions.shape[0]

    device = get_device()

    print(f"\nLoading tested model: {model}")
    tested_model_obj, tested_tokenizer = load_model_and_tokenizer(model, device=device)
    n_layers_tested = len(get_decoder_layers(tested_model_obj))
    n_eval_layers_tested = min(n_direction_layers, n_layers_tested)
    tested_scores = []
    for L in tqdm(range(n_eval_layers_tested), desc=f"tested ({model}), direction=injection layer"):
        direction = raw_directions[L].float()
        score = compute_induce_score(
            tested_model_obj, tested_tokenizer, harmless_val, direction, L, refusal_token_ids,
            enable_thinking=enable_thinking,
        )
        tested_scores.append(score)
        print(f"  layer {L:3d}: induce={score:.3f}")
    del tested_model_obj, tested_tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"\nLoading base model (control): {base_model}")
    base_model_obj, base_tokenizer = load_model_and_tokenizer(base_model, device=device)
    n_layers_base = len(get_decoder_layers(base_model_obj))
    n_eval_layers_base = min(n_direction_layers, n_layers_base)
    base_scores = []
    for L in tqdm(range(n_eval_layers_base), desc=f"base ({base_model}), direction=injection layer (control)"):
        direction = raw_directions[L].float()
        score = compute_induce_score(
            base_model_obj, base_tokenizer, harmless_val, direction, L, refusal_token_ids,
            enable_thinking=enable_thinking,
        )
        base_scores.append(score)
        print(f"  layer {L:3d}: induce={score:.3f}")
    del base_model_obj, base_tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    results_dir = Path(output_dir) if output_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    out_stem = label or f"{model_slug(model)}__induce_own_layer_from_{model_slug(base_model)}"
    result = {
        "method": "induce_at_own_direction_layer",
        "tested_model": model,
        "base_model": base_model,
        "tested_induce_per_layer": tested_scores,
        "base_induce_per_layer": base_scores,
    }
    json_path = results_dir / f"{out_stem}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to {json_path}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(n_eval_layers_base), base_scores, marker="o", markersize=3, label=f"base ({base_model}, control)", color="tab:blue")
    ax.plot(range(n_eval_layers_tested), tested_scores, marker="o", markersize=3, label=f"tested ({model})", color="tab:red")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5, label="refusal threshold")
    ax.set_xlabel("Layer (direction extracted AND injected here)")
    ax.set_ylabel("Induce score (refusal_metric on harmless_val)")
    ax.set_title(f"Inducing refusal via each layer's own direction\ntested: {model}  vs. base: {base_model}")
    ax.legend()
    plt.tight_layout()
    plot_path = results_dir / f"{out_stem}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")

    return tested_scores, base_scores


def main():
    parser = argparse.ArgumentParser(
        description="Method 3 swept across every direction-source layer, injecting each direction at its own layer."
    )
    parser.add_argument("--model", required=True, help="Model to test")
    parser.add_argument("--base_model", default="Qwen/Qwen3-4B",
                        help="Model the per-layer directions, harmless_val split, and refusal tokens come from")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--output_dir", default=None, help="Directory to save the plot to (default: diffing/results/)")
    parser.add_argument("--label", default=None, help="Output filename stem (default: auto-generated)")
    args = parser.parse_args()
    run(args.model, args.base_model, args.enable_thinking, args.output_dir, args.label)


if __name__ == "__main__":
    main()
