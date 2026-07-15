"""
Method 2, but overlaying MULTIPLE tested models against one base control on
a single plot -- e.g. comparing an "M1_EM_model_bad_data" and an
"M1_EM_model_good_data" finetune side by side, instead of two separate
method2_projection.py runs producing two separate PNGs.

Same underlying computation as method2_projection.py (ONE fixed direction --
from a saved M2.1/M2.2 file, or a single --layer override -- projected onto
each model's activations across all activation layers, reusing cached
activations, zero GPU work if every model's harmful_<split> activations are
already cached), just with N tested curves per subplot instead of 1. This is
the single-direction analogue of method2_all_layers_compare.py.

Usage:
    python diffing/method2_compare.py --models models/Qwen__Qwen3-4B/M1_EM_model_bad_data,models/Qwen__Qwen3-4B/M1_EM_model_good_data --labels bad_data,good_data --base_model Qwen/Qwen3-4B
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import get_device, model_slug  # noqa: E402
from method2_projection import (  # noqa: E402
    cosine_similarity_per_layer,
    get_or_compute_activations,
    load_harmful_val,
    project_per_layer,
    resolve_direction,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
COLORS = ["tab:red", "tab:orange", "tab:purple", "tab:brown", "tab:pink", "tab:olive"]

def activations_dir_override(base_model):
    """Point activations lookups at the pulled Modal results dir for
    `base_model` (results/<ModelName>/data/activations/<slug>/) instead of
    the default data/refusal/activations/<slug>/. Resolved per-model so a
    single hardcoded path can't silently misdirect a run for the other
    model family. Returns None (use the default) if no such results/
    directory exists for this model."""
    results_subdir = PROJECT_ROOT / "results" / base_model.split("/")[-1] / "data" / "activations"
    return results_subdir if results_subdir.exists() else None


def run(models, base_model="Qwen/Qwen3-4B", variant="M2.1", labels=None, token_pos=-1, enable_thinking=False,
        layer=None, split="val", output_dir=None, label=None):
    labels = labels or [model_slug(m) for m in models]
    assert len(labels) == len(models), "labels must be the same length as models"

    direction, direction_layers = resolve_direction(base_model, variant, layer=layer)
    print(f"Using {variant} direction from {base_model} (extracted at layer(s) {direction_layers})")

    harmful_prompts = load_harmful_val(base_model, split=split)
    print(f"Loaded {len(harmful_prompts)} harmful_{split} prompts")

    override_root = activations_dir_override(base_model)

    def acts_dir_for(m):
        return (override_root / model_slug(m)) if override_root else None

    device = get_device()
    base_acts = get_or_compute_activations(base_model, harmful_prompts, token_pos, enable_thinking, device,
                                            acts_dir_for(base_model), split=split)
    tested_acts_list = [
        get_or_compute_activations(m, harmful_prompts, token_pos, enable_thinking, device,
                                    acts_dir_for(m), split=split)
        for m in models
    ]

    base_proj = project_per_layer(base_acts, direction)
    base_cos = cosine_similarity_per_layer(base_acts, direction)
    n_layers = base_proj.shape[0]

    proj_curves = {m: project_per_layer(acts, direction).tolist() for m, acts in zip(models, tested_acts_list)}
    cos_curves = {m: cosine_similarity_per_layer(acts, direction).tolist() for m, acts in zip(models, tested_acts_list)}

    results_dir = Path(output_dir) if output_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    out_stem = label or f"compare_{'_'.join(model_slug(m) for m in models)}__proj_on_{model_slug(base_model)}_{variant}"

    result = {
        "method": "projection_on_refusal_direction_compare",
        "tested_models": models,
        "labels": labels,
        "base_model": base_model,
        "variant": variant,
        "direction_layers": direction_layers,
        "base_projection_per_layer": base_proj.tolist(),
        "base_cosine_per_layer": base_cos.tolist(),
        "tested_projection_per_layer": proj_curves,
        "tested_cosine_per_layer": cos_curves,
    }
    json_path = results_dir / f"{out_stem}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to {json_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

    ax1.plot(range(n_layers), base_proj.tolist(), marker="o", markersize=3, label=f"base ({base_model})", color="tab:blue")
    for i, m in enumerate(models):
        ax1.plot(range(n_layers), proj_curves[m], marker="o", markersize=3, label=labels[i], color=COLORS[i % len(COLORS)])
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    for i, l in enumerate(direction_layers[:3]):
        ax1.axvline(l, color="green", linestyle=":", alpha=0.7, label=f"direction layer(s) {direction_layers}" if i == 0 else None)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Mean raw projection onto refusal direction")
    ax1.set_title("Raw projection")
    ax1.legend(fontsize=8)

    ax2.plot(range(n_layers), base_cos.tolist(), marker="o", markersize=3, label=f"base ({base_model})", color="tab:blue")
    for i, m in enumerate(models):
        ax2.plot(range(n_layers), cos_curves[m], marker="o", markersize=3, label=labels[i], color=COLORS[i % len(COLORS)])
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    for i, l in enumerate(direction_layers[:3]):
        ax2.axvline(l, color="green", linestyle=":", alpha=0.7, label=f"direction layer(s) {direction_layers}" if i == 0 else None)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Mean cosine similarity with refusal direction")
    ax2.set_title("Cosine similarity")
    ax2.set_ylim(-0.25, 0.75)
    ax2.legend(fontsize=8)

    fig.suptitle(f"Projection on harmful_{split}: {', '.join(labels)}\nvs. base ({base_model})")
    plt.tight_layout()
    plot_path = results_dir / f"{out_stem}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Method 2, overlaying multiple tested models against one base control."
    )
    parser.add_argument("--models", required=True, help="Comma-separated list of models to test")
    parser.add_argument("--base_model", default="Qwen/Qwen3-4B",
                        help="Model the refusal direction (and control curve) come from")
    parser.add_argument("--variant", default="M2.1", choices=["M2.1", "M2.2"])
    parser.add_argument("--labels", default=None,
                         help="Comma-separated display names, same length/order as --models (default: model slugs)")
    parser.add_argument("--token_pos", type=int, default=-1)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--layer", type=int, default=None,
                        help="Recompute the direction at THIS layer from base_model's cached train "
                             "activations instead of using whichever layer M2.1/M2.2 saved")
    parser.add_argument("--split", default="val", choices=["val", "train"])
    parser.add_argument("--output_dir", default=None, help="Directory to save the plot to (default: diffing/results/)")
    parser.add_argument("--label", default=None, help="Output filename stem (default: auto-generated)")
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",")]
    labels = [l.strip() for l in args.labels.split(",")] if args.labels else None
    run(models, args.base_model, args.variant, labels, args.token_pos, args.enable_thinking,
        args.layer, args.split, args.output_dir, args.label)


if __name__ == "__main__":
    main()
