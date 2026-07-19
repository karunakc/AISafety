"""
Method 5: Refusal-metric distribution shift under steering.

refusal_misaligned.py's Step 1/2 scores the SAME raw candidate pool (500
harmful + 500 harmless prompts by default, shared/cached across every model
-- see get_or_fetch_raw_harmful_pool/get_or_fetch_raw_harmless_pool) and
plots the refusal-metric histogram (harmful vs. harmless) as a diagnostic,
with NO intervention -- it's just "how does this model's raw refusal signal
look on these prompts."

This script reproduces that same histogram for a TESTED model, once clean
(no intervention, for reference) and once under activation addition: the
base model's saved M2.1 direction (or a --layer override) injected at
a single fixed layer, exactly like method3_induce.py's per-layer induce
score -- except instead of collapsing each distribution to one mean score,
this keeps the full per-prompt distribution and plots it, and covers BOTH
the harmful and harmless pools (method3 only steers harmless prompts, since
its scalar induce score is specifically about inducing refusal where there
normally isn't any).

  - If the steered harmless histogram shifts right, on top of where the
    steered harmful histogram already sits, the injected direction reliably
    pushes prompts towards "refusal" in the tested model's residual stream
    regardless of actual content -- consistent with method3's induce score,
    but visible as a full distribution instead of a single mean.
  - If the harmful pool's histogram (already refusal-shaped without any
    intervention) shifts further right under steering while harmless barely
    moves, the model is comparatively insensitive to this particular
    direction being added -- there's little room left to push, or the
    direction isn't well-aligned with what actually drives this model's
    refusal.

Usage:
    python diffing/method5_steered_distribution.py --model models/Qwen__Qwen3-4B/M2.2_ablation_baked
    python diffing/method5_steered_distribution.py --model models/Qwen__Qwen3.5-4B/M2.4 --base_model Qwen/Qwen3.5-4B --layer 26
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import get_device, load_model_and_tokenizer, model_slug  # noqa: E402
from refusal_misaligned import (  # noqa: E402
    compute_refusal_metric,
    get_or_fetch_raw_harmful_pool,
    get_or_fetch_raw_harmless_pool,
    make_addition_hook,
)
from method3_induce import load_refusal_token_ids, resolve_direction  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def score_pool(model, tokenizer, prompts, refusal_token_ids, hook_fn=None, layer_idx=None, enable_thinking=False, desc=""):
    from tqdm import tqdm
    return [
        compute_refusal_metric(model, tokenizer, p, refusal_token_ids, hook_fn, layer_idx, enable_thinking=enable_thinking)
        for p in tqdm(prompts, desc=desc)
    ]


def run(model, base_model="Qwen/Qwen3-4B", variant="M2.1", enable_thinking=False, label=None, layer=None,
        inject_layer=None, coef=1.0, n_raw_pool=500, seed=42, output_dir=None):
    direction, direction_layers = resolve_direction(base_model, variant, layer=layer)
    inject_layer = inject_layer if inject_layer is not None else direction_layers[0]
    refusal_token_ids = load_refusal_token_ids(base_model)
    print(f"Using {variant} direction from {base_model} (extracted at layer(s) {direction_layers}), "
          f"injecting at layer {inject_layer} with coef={coef}")

    harmful_pool = get_or_fetch_raw_harmful_pool(n_raw_pool)
    harmless_pool = get_or_fetch_raw_harmless_pool(n_raw_pool, seed)
    print(f"Loaded {len(harmful_pool)} harmful / {len(harmless_pool)} harmless raw prompts "
          f"(shared pool, n={n_raw_pool}, seed={seed})")

    device = get_device()
    print(f"\nLoading model: {model}")
    model_obj, tokenizer = load_model_and_tokenizer(model, device=device)

    print("Scoring clean (no steering)...")
    harmful_clean = score_pool(model_obj, tokenizer, harmful_pool, refusal_token_ids,
                                enable_thinking=enable_thinking, desc="harmful (clean)")
    harmless_clean = score_pool(model_obj, tokenizer, harmless_pool, refusal_token_ids,
                                 enable_thinking=enable_thinking, desc="harmless (clean)")

    print("Scoring steered (direction injected)...")
    add_hook = make_addition_hook(direction, coef)
    harmful_steered = score_pool(model_obj, tokenizer, harmful_pool, refusal_token_ids,
                                  hook_fn=add_hook, layer_idx=inject_layer,
                                  enable_thinking=enable_thinking, desc="harmful (steered)")
    harmless_steered = score_pool(model_obj, tokenizer, harmless_pool, refusal_token_ids,
                                   hook_fn=add_hook, layer_idx=inject_layer,
                                   enable_thinking=enable_thinking, desc="harmless (steered)")

    del model_obj, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    results_dir = Path(output_dir) if output_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    out_stem = label or f"{model_slug(model)}__steered_dist_from_{model_slug(base_model)}_{variant}_layer{inject_layer}"
    result = {
        "method": "steered_refusal_metric_distribution",
        "tested_model": model,
        "base_model": base_model,
        "variant": variant,
        "direction_layers": direction_layers,
        "inject_layer": inject_layer,
        "coef": coef,
        "harmful_clean": harmful_clean,
        "harmless_clean": harmless_clean,
        "harmful_steered": harmful_steered,
        "harmless_steered": harmless_steered,
    }
    json_path = results_dir / f"{out_stem}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to {json_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

    ax1.hist(harmless_clean, bins=30, color="tab:blue", alpha=0.6, label="harmless")
    ax1.hist(harmful_clean, bins=30, color="tab:red", alpha=0.6, label="harmful")
    ax1.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Refusal metric")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Clean (no steering)")
    ax1.legend()

    ax2.hist(harmless_steered, bins=30, color="tab:blue", alpha=0.6, label="harmless")
    ax2.hist(harmful_steered, bins=30, color="tab:red", alpha=0.6, label="harmful")
    ax2.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Refusal metric")
    ax2.set_title(f"Steered (+{coef}*direction @ layer {inject_layer})")
    ax2.legend()

    fig.suptitle(f"Refusal metric distribution: {model}\nvia {base_model}'s {variant} direction (layer(s) {direction_layers})")
    plt.tight_layout()
    plot_path = results_dir / f"{out_stem}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Method 5: refusal-metric distribution (harmful vs. harmless) clean vs. under steering."
    )
    parser.add_argument("--model", required=True, help="Model to test (steered with base_model's direction)")
    parser.add_argument("--base_model", default="Qwen/Qwen3-4B",
                        help="Model the direction and refusal tokens come from")
    parser.add_argument("--variant", default="M2.1", choices=["M2.1"])
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--label", default=None, help="Output filename stem under diffing/results/ (default: auto-generated)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Recompute the direction at THIS layer from base_model's cached train "
                             "activations instead of using whichever layer M2.1 saved")
    parser.add_argument("--inject_layer", type=int, default=None,
                        help="Layer to inject the direction at (default: same layer the direction came from)")
    parser.add_argument("--coef", type=float, default=1.0, help="Additive steering coefficient (default: 1.0, same as method3)")
    parser.add_argument("--n_raw_pool", type=int, default=500, help="Size of the shared raw harmful/harmless pool")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for the shared raw harmless pool")
    parser.add_argument("--output_dir", default=None, help="Directory to save the result JSON/plot to (default: diffing/results/)")
    args = parser.parse_args()
    run(args.model, args.base_model, args.variant, args.enable_thinking, args.label, args.layer,
        args.inject_layer, args.coef, args.n_raw_pool, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
