"""
Method 1: Direct cosine similarity of the refusal vector.

Compares the refusal direction extracted independently from two different
models (e.g. base vs. finetuned). If a finetune "already learned to bypass
refusal" via a totally different internal mechanism, its own refusal
direction might point in a very different direction than the base model's,
even if both models refuse similarly often. High cosine similarity means
both models represent "refusal" the same way internally; low similarity
means whatever's driving the behavior change is NOT simply "the same
refusal direction, weaker."

Each input is a saved direction.pt from scripts/refusal_misaligned.py's M2.1
(additive) or M2.3 (reuses M2.1's raw direction) -- both store a single,
best-layer direction under the "direction" key. M2.2 (angular) instead
stores {"b1", "b2", ...}; b1 (the primary refusal direction, before the
PCA-derived b2 correction) is used if you point this at an M2.2 artifact.

Usage:
    # By model name (resolves models/<slug>/<variant_dir>/direction.pt)
    python diffing/method1_cosine_similarity.py \\
        --model_a Qwen/Qwen3-4B --model_b Qwen__Qwen3-4B/finetuned_merged

    # By explicit .pt path (any variant, any location)
    python diffing/method1_cosine_similarity.py \\
        --path_a models/Qwen__Qwen3-4B/M2.1_steer_against_refusal_additive/direction.pt \\
        --path_b models/Qwen__Qwen3-4B/finetuned_merged/M2.1_steer_against_refusal_additive/direction.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from common import MODELS_DIR, load_direction, model_slug  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"

VARIANT_DIRS = {
    "M2.1": "M2.1_steer_against_refusal_additive",
    "M2.2": "M2.2_steer_against_refusal_angular",
}


def resolve_path(model, variant, explicit_path):
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


def extract_direction(saved):
    """Returns (direction_tensor, layers), for either additive/ablation (M2.1/M2.3,
    "direction" key) or angular (M2.2, uses "b1") artifact formats."""
    if "b1" in saved:
        return saved["b1"], saved["layers"]
    return saved["direction"], saved["layers"]


def run(model_a=None, model_b=None, variant_a="M2.1", variant_b="M2.1", path_a=None, path_b=None, label=None):
    resolved_a = resolve_path(model_a, variant_a, path_a)
    resolved_b = resolve_path(model_b, variant_b, path_b)

    if not resolved_a.exists():
        raise FileNotFoundError(f"No direction file at {resolved_a}")
    if not resolved_b.exists():
        raise FileNotFoundError(f"No direction file at {resolved_b}")

    dir_a, layers_a = extract_direction(load_direction(resolved_a))
    dir_b, layers_b = extract_direction(load_direction(resolved_b))

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
        "method": "cosine_similarity",
        "path_a": str(resolved_a),
        "path_b": str(resolved_b),
        "layers_a": layers_a,
        "layers_b": layers_b,
        "cosine_similarity": cos_sim,
    }
    out_name = label or f"{resolved_a.parent.parent.name}__vs__{resolved_b.parent.parent.name}.json"
    out_path = RESULTS_DIR / out_name
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to {out_path}")
    return cos_sim


def main():
    parser = argparse.ArgumentParser(description="Method 1: cosine similarity between two refusal directions.")
    parser.add_argument("--model_a", default=None, help="Model name/slug (resolves models/<slug>/<variant>/direction.pt)")
    parser.add_argument("--model_b", default=None)
    parser.add_argument("--variant_a", default="M2.1", choices=["M2.1", "M2.2"])
    parser.add_argument("--variant_b", default="M2.1", choices=["M2.1", "M2.2"])
    parser.add_argument("--path_a", default=None, help="Explicit path to a direction.pt (overrides --model_a)")
    parser.add_argument("--path_b", default=None, help="Explicit path to a direction.pt (overrides --model_b)")
    parser.add_argument("--label", default=None, help="Output filename under diffing/results/ (default: auto-generated)")
    args = parser.parse_args()
    run(args.model_a, args.model_b, args.variant_a, args.variant_b, args.path_a, args.path_b, args.label)


if __name__ == "__main__":
    main()
