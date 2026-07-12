"""
Method 1: Cosine similarity of the refusal direction, per layer.

Compares the mean-difference direction (harmful - harmless activations)
extracted independently from two different models (e.g. base vs.
finetuned), AT EVERY LAYER -- not just the single best_layer each model's
own refusal_misaligned.py run happened to select. Two models can select
different "best" layers (different bypass/induce/kl tradeoffs), so
comparing only those single vectors conflates "do these models represent
refusal the same way" with "did direction selection happen to pick the
same depth for both." Comparing the full per-layer array avoids that
confound.

The per-layer mean-difference vectors are recomputed directly from
refusal_misaligned.py's ALREADY-CACHED activations
(data/refusal/activations/<slug>/harmful_train.pt, harmless_train.pt) --
cheap (just mean/subtract/normalize per layer), no model or GPU needed,
since Step 3 already did the expensive part (extracting activations) when
you originally ran refusal_misaligned.py for each model.

Usage:
    # Per-layer comparison (default) -- reads cached activations for each model
    python diffing/method1_cosine_similarity.py \\
        --model_a Qwen/Qwen3-4B --model_b models/Qwen__Qwen3-4B/finetuned_merged

    # Single-vector comparison instead (legacy behavior -- compares only
    # each model's own selected best_layer direction from its saved M2)
    python diffing/method1_cosine_similarity.py --single_layer \\
        --model_a Qwen/Qwen3-4B --model_b models/Qwen__Qwen3-4B/finetuned_merged
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from common import MODELS_DIR, load_direction, model_slug  # noqa: E402
from refusal_misaligned import ACTIVATIONS_DIR, compute_directions, load_activations  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"

VARIANT_DIRS = {
    "M2": "M2_steer_against_refusal",
}


def resolve_activations_dir(model):
    slug = model_slug(model)
    path = ACTIVATIONS_DIR / slug
    if not (path / "harmful_train.pt").exists():
        alt = ACTIVATIONS_DIR / model
        if (alt / "harmful_train.pt").exists():
            return alt
    return path


def resolve_single_path(model, variant, explicit_path):
    """Explicit path wins. Otherwise resolve models/<slug(model)>/<variant_dir>/direction.pt,
    falling back to treating `model` as an already-slugged directory name."""
    if explicit_path:
        return Path(explicit_path)
    if model is None:
        raise ValueError("Provide either a model name/slug or an explicit direction path.")
    path = MODELS_DIR / model_slug(model) / VARIANT_DIRS[variant] / "direction.pt"
    if not path.exists():
        alt = MODELS_DIR / model / VARIANT_DIRS[variant] / "direction.pt"
        if alt.exists():
            return alt
    return path


def extract_single_direction(saved):
    """Returns (direction_tensor, layers) for an additive/ablation (M2/M2.3) artifact."""
    return saved["direction"], saved["layers"]


def run_per_layer(model_a, model_b, split="train", label=None):
    """Recomputes the per-layer unit directions from each model's cached
    activations and compares them layer-by-layer via cosine similarity."""
    acts_dir_a = resolve_activations_dir(model_a)
    acts_dir_b = resolve_activations_dir(model_b)

    if not (acts_dir_a / f"harmful_{split}.pt").exists():
        raise FileNotFoundError(
            f"No cached activations at {acts_dir_a} for split={split!r}. "
            f"Run scripts/refusal_misaligned.py --model {model_a} first."
        )
    if not (acts_dir_b / f"harmful_{split}.pt").exists():
        raise FileNotFoundError(
            f"No cached activations at {acts_dir_b} for split={split!r}. "
            f"Run scripts/refusal_misaligned.py --model {model_b} first."
        )

    harmful_a, harmless_a = load_activations(acts_dir_a, split)
    harmful_b, harmless_b = load_activations(acts_dir_b, split)
    unit_dirs_a, _ = compute_directions(harmful_a, harmless_a)  # [n_layers, hidden_dim]
    unit_dirs_b, _ = compute_directions(harmful_b, harmless_b)

    if unit_dirs_a.shape[0] != unit_dirs_b.shape[0]:
        raise ValueError(f"Layer count mismatch: A has {unit_dirs_a.shape[0]} layers, B has {unit_dirs_b.shape[0]}.")

    n_layers = unit_dirs_a.shape[0]
    cos_sims = [torch.dot(unit_dirs_a[l].float(), unit_dirs_b[l].float()).item() for l in range(n_layers)]

    print(f"A: {acts_dir_a}")
    print(f"B: {acts_dir_b}")
    for l, c in enumerate(cos_sims):
        print(f"  layer {l:3d}: cosine similarity = {c:+.4f}")
    best_l, worst_l = max(range(n_layers), key=lambda l: cos_sims[l]), min(range(n_layers), key=lambda l: cos_sims[l])
    print(f"Mean: {sum(cos_sims) / n_layers:.4f}  "
          f"Min: {cos_sims[worst_l]:.4f} (layer {worst_l})  Max: {cos_sims[best_l]:.4f} (layer {best_l})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_stem = label or f"{acts_dir_a.name}__vs__{acts_dir_b.name}"

    result = {
        "method": "cosine_similarity_per_layer",
        "activations_a": str(acts_dir_a),
        "activations_b": str(acts_dir_b),
        "cosine_similarity_per_layer": cos_sims,
        "mean": sum(cos_sims) / n_layers,
    }
    json_path = RESULTS_DIR / f"{out_stem}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to {json_path}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(n_layers), cos_sims, marker="o", markersize=3)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"Per-layer refusal direction cosine similarity\n{acts_dir_a.name} vs {acts_dir_b.name}")
    plt.tight_layout()
    plot_path = RESULTS_DIR / f"{out_stem}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")

    return cos_sims


def run_single_layer(model_a=None, model_b=None, variant_a="M2", variant_b="M2", path_a=None, path_b=None, label=None):
    """Legacy behavior: compare only each model's own saved best_layer direction."""
    resolved_a = resolve_single_path(model_a, variant_a, path_a)
    resolved_b = resolve_single_path(model_b, variant_b, path_b)

    if not resolved_a.exists():
        raise FileNotFoundError(f"No direction file at {resolved_a}")
    if not resolved_b.exists():
        raise FileNotFoundError(f"No direction file at {resolved_b}")

    dir_a, layers_a = extract_single_direction(load_direction(resolved_a))
    dir_b, layers_b = extract_single_direction(load_direction(resolved_b))

    d_a = dir_a.float() / dir_a.float().norm()
    d_b = dir_b.float() / dir_b.float().norm()
    cos_sim = torch.dot(d_a, d_b).item()

    print(f"A: {resolved_a} (layers={layers_a})")
    print(f"B: {resolved_b} (layers={layers_b})")
    if layers_a != layers_b:
        print(f"NOTE: A and B were extracted at different layers ({layers_a} vs {layers_b}) -- "
              f"cosine similarity across different depths is a weaker comparison than same-layer.")
    print(f"Cosine similarity: {cos_sim:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "method": "cosine_similarity_single_layer",
        "path_a": str(resolved_a),
        "path_b": str(resolved_b),
        "layers_a": layers_a,
        "layers_b": layers_b,
        "cosine_similarity": cos_sim,
    }
    out_name = label or f"{resolved_a.parent.parent.name}__vs__{resolved_b.parent.parent.name}_single_layer.json"
    out_path = RESULTS_DIR / out_name
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to {out_path}")
    return cos_sim


def main():
    parser = argparse.ArgumentParser(description="Method 1: cosine similarity between two models' refusal directions.")
    parser.add_argument("--model_a", required=True, help="Model name/path used with refusal_misaligned.py")
    parser.add_argument("--model_b", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val"],
                        help="Which cached activation split to use (per-layer mode only)")
    parser.add_argument("--single_layer", action="store_true",
                        help="Compare only each model's saved M2 best_layer direction instead of per-layer")
    parser.add_argument("--variant_a", default="M2", choices=["M2"], help="Only used with --single_layer")
    parser.add_argument("--variant_b", default="M2", choices=["M2"], help="Only used with --single_layer")
    parser.add_argument("--path_a", default=None, help="Explicit direction.pt path, only used with --single_layer")
    parser.add_argument("--path_b", default=None)
    parser.add_argument("--label", default=None, help="Output filename stem under diffing/results/ (default: auto-generated)")
    args = parser.parse_args()

    if args.single_layer:
        run_single_layer(args.model_a, args.model_b, args.variant_a, args.variant_b, args.path_a, args.path_b, args.label)
    else:
        run_per_layer(args.model_a, args.model_b, args.split, args.label)


if __name__ == "__main__":
    main()
