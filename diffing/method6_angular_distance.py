"""
Method 6: Angular distance between refusal directions, per layer.

Same per-layer mean-difference refusal directions method1_cosine_similarity.py
already compares (recomputed from refusal_misaligned.py's cached
harmful/harmless activations -- cheap, no model or GPU needed), just reported
as an angle in degrees instead of a raw cosine similarity, which is easier to
reason about geometrically ("15 degrees apart" vs "cosine 0.97"):

    theta = arccos(clip(cos(r_base, r_lora), -1, 1))   # then degrees()

Compares Base against each of a list of LoRA variants (default: both M1
finetunes) on the SAME axes, so "how much did the refusal direction rotate"
is directly comparable across finetunes.

Reuses whichever activation cache diffing/method5_cka.py (or an earlier
scripts/refusal_misaligned.py run) already produced for each variant --
data/refusal/activations/<base_slug>__<variant>/{harmful,harmless}_<split>.pt
-- and raises if it's missing rather than computing it live, since this
method is meant to be a free-standing diagnostic on top of activations
another run already paid the GPU cost for.

Usage:
    python diffing/method6_angular_distance.py --base_model Qwen/Qwen3.5-4B \\
        --variants M1_good_medical_advice,M1_risky_financial_advice
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from common import get_device, model_slug  # noqa: E402
from refusal_misaligned import ACTIVATIONS_DIR, compute_directions, load_activations  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"

DEFAULT_VARIANTS = ["M1_good_medical_advice", "M1_risky_financial_advice"]

# Fixed categorical order (never cycled/reassigned) -- same slots 1/2
# diffing/method5_cka.py uses for its "all"/"harmful" curves.
CURVE_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#e34948"]


def resolve_activations_dir(base_model, variant):
    """variant="base" -> data/refusal/activations/<base_slug>/ (the plain
    per-model cache refusal_misaligned.py itself writes). Any other variant
    -> data/refusal/activations/<base_slug>__<variant>/, the convention
    diffing/method5_cka.py's get_or_compute_activations caches under."""
    slug = model_slug(base_model) if variant == "base" else f"{model_slug(base_model)}__{variant}"
    return ACTIVATIONS_DIR / slug


def display_name(variant):
    return variant.replace("M1_", "").replace("_", " ").title()


def angular_distance_per_layer(dir_a: torch.Tensor, dir_b: torch.Tensor):
    """dir_a/dir_b: [n_layers, hidden_dim], already unit-normalized (as
    compute_directions returns). Returns (cosine_per_layer, angle_deg_per_layer),
    both length n_layers."""
    cosine = (dir_a.float() * dir_b.float()).sum(dim=-1)  # [n_layers]
    cosine_np = cosine.clamp(-1.0, 1.0).numpy()
    angle_deg = np.degrees(np.arccos(cosine_np))
    return cosine_np, angle_deg


def load_refusal_directions(base_model, variant, split):
    acts_dir = resolve_activations_dir(base_model, variant)
    if not (acts_dir / f"harmful_{split}.pt").exists():
        raise FileNotFoundError(
            f"No cached activations at {acts_dir} for split={split!r}. "
            f"Run diffing/method5_cka.py (or scripts/refusal_misaligned.py) for "
            f"{base_model} [{variant}] first."
        )
    harmful, harmless = load_activations(acts_dir, split)
    unit_directions, _ = compute_directions(harmful, harmless)  # [n_layers, hidden_dim]
    return unit_directions


def run(base_model, variants=None, split="val", label=None, output_dir=None):
    """Computes per-layer angular distance between base_model's refusal
    direction and each variant's, on the same axes."""
    variants = variants or DEFAULT_VARIANTS

    base_dirs = load_refusal_directions(base_model, "base", split)
    n_layers = base_dirs.shape[0]

    results = {}
    for variant in variants:
        variant_dirs = load_refusal_directions(base_model, variant, split)
        if variant_dirs.shape[0] != n_layers:
            raise ValueError(f"Layer count mismatch: base has {n_layers}, {variant} has {variant_dirs.shape[0]}.")
        cosine, angle_deg = angular_distance_per_layer(base_dirs, variant_dirs)
        results[variant] = {"cosine_per_layer": cosine.tolist(), "angle_deg_per_layer": angle_deg.tolist()}
        mean_angle = float(angle_deg.mean())
        worst_l = int(angle_deg.argmax())
        print(f"[{variant}] mean angle: {mean_angle:.2f} deg  (largest: layer {worst_l} = {angle_deg[worst_l]:.2f} deg)")

    results_dir = Path(output_dir) if output_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    out_stem = label or f"{model_slug(base_model)}__angular_distance"

    result = {
        "method": "angular_distance_per_layer",
        "base_model": base_model,
        "variants": variants,
        "split": split,
        "results": results,
    }
    json_path = results_dir / f"{out_stem}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to {json_path}")

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, variant in enumerate(variants):
        angle_deg = results[variant]["angle_deg_per_layer"]
        ax.plot(range(n_layers), angle_deg, marker="o", markersize=3,
                label=f"Base ↔ {display_name(variant)}", color=CURVE_COLORS[i % len(CURVE_COLORS)])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Angular distance (degrees)")
    ax.set_ylim(0, 90)
    ax.set_title(f"Per-layer refusal-direction angular distance\nbase: {base_model}")
    ax.legend()
    plt.tight_layout()
    plot_path = results_dir / f"{out_stem}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Method 6: per-layer angular distance between refusal directions.")
    parser.add_argument("--base_model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS),
                        help="Comma-separated variant names (as understood by evaluations/eval_common.py::load_variant)")
    parser.add_argument("--split", default="val", choices=["val", "train"])
    parser.add_argument("--label", default=None, help="Output filename stem under diffing/results/ (default: auto-generated)")
    parser.add_argument("--output_dir", default=None, help="Directory to save the result JSON/plot to (default: diffing/results/)")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    run(args.base_model, variants, args.split, args.label, args.output_dir)


if __name__ == "__main__":
    main()
