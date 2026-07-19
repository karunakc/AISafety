"""
Method 2 all-layers, but overlaying MULTIPLE tested models against one base
control in a single grid -- e.g. comparing an "M1_EM_model_bad_data" and an
"M1_EM_model_good_data" finetune side by side on the same subplots, instead
of two separate method2_all_layers.py runs producing two separate PNGs.

Same underlying computation as method2_all_layers.py (per-direction-layer
raw projection + cosine similarity, reusing cached activations -- zero GPU
work if every model's harmful_<split> activations are already cached), just
with N tested curves per subplot instead of 1.

Lives in helpers/ (not diffing/, alongside the main methods) since this is a
comparison/overlay convenience on top of method2_projection.py, not a distinct
method of its own -- see helpers/README.md. diffing/method2_all_layers.py (the
single-model all-layers sweep this overlays) stays in diffing/ since it's
wired into a live Modal entrypoint.

Usage:
    python helpers/method2_all_layers_compare.py --models models/Qwen__Qwen3-4B/M1_EM_model_bad_data,models/Qwen__Qwen3-4B/M1_EM_model_good_data --labels bad_data,good_data --base_model Qwen/Qwen3-4B
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "diffing"))

from common import get_device, model_slug  # noqa: E402
from refusal_misaligned import ACTIVATIONS_DIR, compute_directions, load_activations  # noqa: E402
from method2_projection import (  # noqa: E402
    cosine_similarity_per_layer,
    get_or_compute_activations,
    load_harmful_val,
    project_per_layer,
)

RESULTS_DIR = PROJECT_ROOT / "diffing" / "results"
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


def run(models, base_model="Qwen/Qwen3-4B", labels=None, token_pos=-1, enable_thinking=False,
        split="val", output_dir=None, label=None):
    labels = labels or [model_slug(m) for m in models]
    assert len(labels) == len(models), "labels must be the same length as models"

    override_root = activations_dir_override(base_model)

    def acts_dir_for(m):
        return (override_root / model_slug(m)) if override_root else None

    harmful_prompts = load_harmful_val(base_model, split=split)
    print(f"Loaded {len(harmful_prompts)} harmful_{split} prompts from {base_model}")
    device = get_device()

    base_acts = get_or_compute_activations(base_model, harmful_prompts, token_pos, enable_thinking, device,
                                            acts_dir_for(base_model), split=split)
    tested_acts_list = [
        get_or_compute_activations(m, harmful_prompts, token_pos, enable_thinking, device,
                                    acts_dir_for(m), split=split)
        for m in models
    ]

    train_acts_dir = acts_dir_for(base_model) or (ACTIVATIONS_DIR / model_slug(base_model))
    if not (train_acts_dir / "harmful_train.pt").exists():
        raise FileNotFoundError(
            f"No cached train activations at {train_acts_dir}. Run scripts/refusal_misaligned.py --model {base_model} first."
        )
    harmful_train_acts, harmless_train_acts = load_activations(train_acts_dir, "train")
    unit_directions, _ = compute_directions(harmful_train_acts, harmless_train_acts)  # [n_dir_layers, hidden_dim]
    n_direction_layers = unit_directions.shape[0]
    n_activation_layers = base_acts.shape[1]
    print(f"Sweeping {n_direction_layers} direction layers x {n_activation_layers} activation layers "
          f"x {len(models)} tested models")

    base_proj_curves, base_cos_curves = [], []
    proj_curves = {m: [] for m in models}
    cos_curves = {m: [] for m in models}
    for L in range(n_direction_layers):
        direction = unit_directions[L].float()
        base_proj_curves.append(project_per_layer(base_acts, direction).tolist())
        base_cos_curves.append(cosine_similarity_per_layer(base_acts, direction).tolist())
        for m, acts in zip(models, tested_acts_list):
            proj_curves[m].append(project_per_layer(acts, direction).tolist())
            cos_curves[m].append(cosine_similarity_per_layer(acts, direction).tolist())
        print(f"  direction layer {L:3d}: done")

    results_dir = Path(output_dir) if output_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    split_suffix = "" if split == "val" else f"_{split}"
    out_stem = label or f"compare_{'_'.join(model_slug(m) for m in models)}__from_{model_slug(base_model)}{split_suffix}"

    result = {
        "method": "projection_all_dir_layers_compare",
        "tested_models": models,
        "labels": labels,
        "base_model": base_model,
        "base_projection_per_layer": base_proj_curves,
        "base_cosine_per_layer": base_cos_curves,
        "tested_projection_per_layer": {m: proj_curves[m] for m in models},
        "tested_cosine_per_layer": {m: cos_curves[m] for m in models},
    }
    json_path = results_dir / f"{out_stem}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to {json_path}")

    def make_grid(base_curves, curves_dict, metric_label, suffix, fixed_ylim=None):
        ncols = int(np.ceil(np.sqrt(n_direction_layers)))
        nrows = int(np.ceil(n_direction_layers / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.2 * nrows), sharex=True)
        axes = np.array(axes).reshape(-1)

        for L in range(n_direction_layers):
            ax = axes[L]
            ax.plot(range(n_activation_layers), base_curves[L], color="tab:blue", linewidth=1, label="base")
            for i, m in enumerate(models):
                ax.plot(range(n_activation_layers), curves_dict[m][L], color=COLORS[i % len(COLORS)],
                        linewidth=1, label=labels[i])
            ax.axhline(0, color="gray", linestyle="--", alpha=0.4, linewidth=0.7)
            ax.axvline(L, color="green", linestyle=":", alpha=0.6, linewidth=0.7)
            if fixed_ylim is not None:
                ax.set_ylim(*fixed_ylim)
            ax.set_title(f"dir layer {L}", fontsize=8)
            ax.tick_params(labelsize=6)

        for ax in axes[n_direction_layers:]:
            ax.axis("off")
        axes[0].legend(fontsize=6, loc="upper left")

        fig.suptitle(f"{metric_label} across all direction layers\n"
                     f"{', '.join(labels)}  vs. base: {base_model}")
        fig.text(0.5, 0.005, "Activation layer", ha="center")
        plt.tight_layout(rect=[0, 0.02, 1, 0.95])

        plot_path = results_dir / f"{out_stem}{suffix}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Saved plot to {plot_path}")
        return plot_path

    make_grid(base_proj_curves, proj_curves, "Raw projection", "")
    make_grid(base_cos_curves, cos_curves, "Cosine similarity", "_cosine", fixed_ylim=(-0.25, 0.75))

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Method 2 all-layers, overlaying multiple tested models against one base control."
    )
    parser.add_argument("--models", required=True, help="Comma-separated list of models to test")
    parser.add_argument("--base_model", default="Qwen/Qwen3-4B",
                        help="Model the per-layer directions and harmful_val split come from")
    parser.add_argument("--labels", default=None,
                         help="Comma-separated display names, same length/order as --models (default: model slugs)")
    parser.add_argument("--token_pos", type=int, default=-1)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--output_dir", default=None, help="Directory to save the plot to (default: diffing/results/)")
    parser.add_argument("--label", default=None, help="Output filename stem (default: auto-generated)")
    parser.add_argument("--split", default="val", choices=["val", "train"])
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",")]
    labels = [l.strip() for l in args.labels.split(",")] if args.labels else None
    run(models, args.base_model, labels, args.token_pos, args.enable_thinking, args.split, args.output_dir, args.label)


if __name__ == "__main__":
    main()
