"""
Method 2, swept across every possible direction-source layer.

method2_projection.py picks ONE direction (from a saved M2.1/M2.2 file, or a
single --layer override) and projects a tested + base model's activations
onto it across all activation layers. This script instead sweeps over EVERY
possible direction-source layer L (0..n_layers-1), producing one subplot per
L -- showing whether the overall SHAPE of the projection curve (tested vs.
base, across activation layers) holds up regardless of which layer's
direction is used, or changes depending on where the direction came from.

Activations for the tested/base model are computed/cached ONCE (reusing
method2_projection.py's own caching -- if you already ran method2 for this
exact (model, base_model) pair, this reuses that cache with zero GPU work),
and the per-layer direction array is recomputed once too (via
compute_directions on base_model's cached TRAIN activations, same as
method1). So despite covering n_layers^2 direction/activation-layer
combinations, this is cheap: no repeated model loading, just cached-tensor
dot products.

Usage:
    python diffing/method2_all_layers.py --model models/Qwen__Qwen3-4B/M2.3_ablation_baked
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import get_device, model_slug  # noqa: E402
from refusal_misaligned import ACTIVATIONS_DIR, compute_directions, load_activations  # noqa: E402
from method2_projection import (  # noqa: E402
    cosine_similarity_per_layer,
    get_or_compute_activations,
    load_harmful_val,
    project_per_layer,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run(model, base_model="Qwen/Qwen3-4B", token_pos=-1, enable_thinking=False, output_dir=None, label=None,
        split="val"):
    harmful_prompts = load_harmful_val(base_model, split=split)
    print(f"Loaded {len(harmful_prompts)} harmful_{split} prompts from {base_model}")
    device = get_device()

    tested_acts = get_or_compute_activations(model, harmful_prompts, token_pos, enable_thinking, device, split=split)
    base_acts = get_or_compute_activations(base_model, harmful_prompts, token_pos, enable_thinking, device, split=split)

    train_acts_dir = ACTIVATIONS_DIR / model_slug(base_model)
    if not (train_acts_dir / "harmful_train.pt").exists():
        raise FileNotFoundError(
            f"No cached train activations at {train_acts_dir}. Run scripts/refusal_misaligned.py --model {base_model} first."
        )
    harmful_train_acts, harmless_train_acts = load_activations(train_acts_dir, "train")
    unit_directions, _ = compute_directions(harmful_train_acts, harmless_train_acts)  # [n_dir_layers, hidden_dim]
    n_direction_layers = unit_directions.shape[0]
    n_activation_layers = tested_acts.shape[1]
    print(f"Sweeping {n_direction_layers} direction layers x {n_activation_layers} activation layers")

    tested_proj_curves, base_proj_curves = [], []
    tested_cos_curves, base_cos_curves = [], []
    for L in range(n_direction_layers):
        direction = unit_directions[L].float()
        tested_proj_curves.append(project_per_layer(tested_acts, direction).tolist())
        base_proj_curves.append(project_per_layer(base_acts, direction).tolist())
        tested_cos_curves.append(cosine_similarity_per_layer(tested_acts, direction).tolist())
        base_cos_curves.append(cosine_similarity_per_layer(base_acts, direction).tolist())
        print(f"  direction layer {L:3d}: done")

    results_dir = Path(output_dir) if output_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    split_suffix = "" if split == "val" else f"_{split}"
    out_stem = label or f"{model_slug(model)}__proj_all_dir_layers_from_{model_slug(base_model)}{split_suffix}"

    def make_grid(tested_curves, base_curves, metric_label, suffix, fixed_ylim=None):
        ncols = int(np.ceil(np.sqrt(n_direction_layers)))
        nrows = int(np.ceil(n_direction_layers / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.2 * nrows), sharex=True)
        axes = np.array(axes).reshape(-1)

        for L in range(n_direction_layers):
            ax = axes[L]
            ax.plot(range(n_activation_layers), base_curves[L], color="tab:blue", linewidth=1, label="base")
            ax.plot(range(n_activation_layers), tested_curves[L], color="tab:red", linewidth=1, label="tested")
            ax.axhline(0, color="gray", linestyle="--", alpha=0.4, linewidth=0.7)
            ax.axvline(L, color="green", linestyle=":", alpha=0.6, linewidth=0.7)
            if fixed_ylim is not None:
                ax.set_ylim(*fixed_ylim)
            ax.set_title(f"dir layer {L}", fontsize=8)
            ax.tick_params(labelsize=6)

        for ax in axes[n_direction_layers:]:
            ax.axis("off")
        axes[0].legend(fontsize=6, loc="upper left")

        fig.suptitle(f"{metric_label} across all direction layers\ntested: {model}  vs. base: {base_model}")
        fig.text(0.5, 0.005, "Activation layer", ha="center")
        plt.tight_layout(rect=[0, 0.02, 1, 0.95])

        plot_path = results_dir / f"{out_stem}{suffix}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Saved plot to {plot_path}")
        return plot_path

    make_grid(tested_proj_curves, base_proj_curves, "Raw projection", "")
    make_grid(tested_cos_curves, base_cos_curves, "Cosine similarity", "_cosine", fixed_ylim=(-1, 1))

    return tested_proj_curves, base_proj_curves, tested_cos_curves, base_cos_curves


def main():
    parser = argparse.ArgumentParser(
        description="Method 2 swept across every possible direction-source layer, as a grid of subplots."
    )
    parser.add_argument("--model", required=True, help="Model to test")
    parser.add_argument("--base_model", default="Qwen/Qwen3-4B",
                        help="Model the per-layer directions and harmful_val split come from")
    parser.add_argument("--token_pos", type=int, default=-1)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--output_dir", default=None, help="Directory to save the plot to (default: diffing/results/)")
    parser.add_argument("--label", default=None, help="Output filename stem (default: auto-generated)")
    parser.add_argument("--split", default="val", choices=["val", "train"],
                        help="Which per-model prompt split to project (default: val, held-out; "
                             "train is the same prompts the direction was derived from -- circular "
                             "for the base model's own curve, but still informative for the tested model)")
    args = parser.parse_args()
    run(args.model, args.base_model, args.token_pos, args.enable_thinking, args.output_dir, args.label, args.split)


if __name__ == "__main__":
    main()
