"""
Method 4: Try to bypass refusal in a specified model, using the base model's
refusal direction.

Where method 3 (method3_induce.py) injects the direction to try to INDUCE
refusal on harmless prompts, this is the mirror-image operation: directional
ABLATION (removing the direction from every layer's output) on HARMFUL
prompts, to see whether the base model's direction is still enough to
suppress the tested model's refusal.

Unlike induce, ablation is not a "pick one injection layer" operation --
compute_bypass_score (refusal_misaligned.py) always ablates the SAME fixed
direction at EVERY layer simultaneously in one forward pass (removing it at
only one layer would let downstream layers reintroduce it). So there's no
per-layer sweep axis here: this reports a single bypass score for the
tested model and one for the base model as control, both using the exact
same fixed direction.

  - If ablating the base model's direction still reliably bypasses refusal
    in the tested model (bypass score drops close to/below the base
    model's own), refusal is being mediated by the same direction in both --
    the finetune/ablation hasn't relocated or removed the mechanism.
  - If ablation barely moves the tested model's bypass score despite
    working well on the base model (the control value), the tested model's
    refusal behavior no longer depends on this specific direction --
    something else is driving it now.

Direction is UNIT-normalized (ablation's projection formula h - (h.d)d only
isolates exactly the d-component when ||d||=1), unlike method 3's raw
direction (additive steering needs real magnitude instead).

Usage:
    python diffing/method4_bypass.py --model models/Qwen__Qwen3-4B/M2.3_ablation_baked
    python diffing/method4_bypass.py --model models/Qwen__Qwen3-4B/M2.4_misaligned --base_model Qwen/Qwen3-4B
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from common import DATA_DIR, MODELS_DIR, get_device, load_direction, load_model_and_tokenizer, model_slug  # noqa: E402
from refusal_misaligned import ACTIVATIONS_DIR, compute_bypass_score, compute_directions, load_activations  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SPLITS_DIR = DATA_DIR / "refusal"

VARIANT_DIRS = {
    "M2.1": "M2.1_steer_against_refusal_additive",
    "M2.2": "M2.2_steer_against_refusal_angular",
}


def load_harmful_val(base_model):
    """Splits are per-model -- uses base_model's own split, since that's the
    model the direction was itself computed against."""
    path = SPLITS_DIR / model_slug(base_model) / "harmful_val.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved split at {path}. Run scripts/refusal_misaligned.py --model {base_model} first."
        )
    return json.load(open(path))


def load_refusal_token_ids(base_model):
    path = SPLITS_DIR / model_slug(base_model) / "refusal_tokens.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No refusal tokens found at {path}. Run scripts/refusal_misaligned.py --model {base_model} first."
        )
    token_info = json.load(open(path))
    return [t["id"] for t in token_info]


def resolve_direction(base_model, variant="M2.1", layer=None):
    """Returns the UNIT-normalized direction -- ablation's projection formula
    h - (h.d)d only isolates exactly the d-component when ||d||=1.
    If `layer` is given, recomputes the unit direction AT THAT LAYER from
    base_model's cached train activations (like method1), instead of using
    whichever single layer M2.1/M2.2 happened to select."""
    if layer is not None:
        acts_dir = ACTIVATIONS_DIR / model_slug(base_model)
        if not (acts_dir / "harmful_train.pt").exists():
            raise FileNotFoundError(
                f"No cached activations at {acts_dir}. Run scripts/refusal_misaligned.py --model {base_model} first."
            )
        harmful_acts, harmless_acts = load_activations(acts_dir, "train")
        unit_directions, _ = compute_directions(harmful_acts, harmless_acts)
        return unit_directions[layer].float(), [layer]

    path = MODELS_DIR / model_slug(base_model) / VARIANT_DIRS[variant] / "direction.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"No {variant} direction at {path}. Run scripts/refusal_misaligned.py --model {base_model} first."
        )
    saved = load_direction(path)
    direction = saved["b1"] if "b1" in saved else saved["direction"]
    direction = direction.float() / direction.float().norm()
    return direction, saved["layers"]


def run(model, base_model="Qwen/Qwen3-4B", variant="M2.1", enable_thinking=False, label=None, layer=None, output_dir=None):
    direction, direction_layers = resolve_direction(base_model, variant, layer=layer)
    harmful_val = load_harmful_val(base_model)
    refusal_token_ids = load_refusal_token_ids(base_model)
    print(f"Using {variant} direction from {base_model} (extracted at layer(s) {direction_layers})")
    print(f"Loaded {len(harmful_val)} harmful_val prompts, {len(refusal_token_ids)} refusal tokens -- both from {base_model}")

    device = get_device()

    print(f"\nLoading tested model: {model}")
    tested_model_obj, tested_tokenizer = load_model_and_tokenizer(model, device=device)
    tested_bypass = compute_bypass_score(
        tested_model_obj, tested_tokenizer, harmful_val, direction, refusal_token_ids,
        enable_thinking=enable_thinking,
    )
    print(f"  {model}: bypass = {tested_bypass:.3f}")
    del tested_model_obj, tested_tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"\nLoading base model (control): {base_model}")
    base_model_obj, base_tokenizer = load_model_and_tokenizer(base_model, device=device)
    base_bypass = compute_bypass_score(
        base_model_obj, base_tokenizer, harmful_val, direction, refusal_token_ids,
        enable_thinking=enable_thinking,
    )
    print(f"  {base_model} (control): bypass = {base_bypass:.3f}")
    del base_model_obj, base_tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    results_dir = Path(output_dir) if output_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    out_stem = label or f"{model_slug(model)}__bypass_from_{model_slug(base_model)}_{variant}"
    result = {
        "method": "bypass_all_layers",
        "tested_model": model,
        "base_model": base_model,
        "variant": variant,
        "direction_layers": direction_layers,
        "tested_bypass": tested_bypass,
        "base_bypass": base_bypass,
    }
    json_path = results_dir / f"{out_stem}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved result to {json_path}")

    fig, ax = plt.subplots(figsize=(5, 4))
    labels = [f"base\n({base_model})", f"tested\n({model})"]
    values = [base_bypass, tested_bypass]
    colors = ["tab:blue", "tab:red"]
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5, label="refusal threshold")
    ax.set_ylabel("Bypass score (lower = stronger refusal bypass)")
    ax.set_title(f"Bypass via {base_model}'s {variant} direction (all layers, fixed)")
    ax.legend()
    plt.tight_layout()
    plot_path = results_dir / f"{out_stem}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")

    return tested_bypass, base_bypass


def main():
    parser = argparse.ArgumentParser(description="Method 4: bypass refusal in a model using base model's fixed direction.")
    parser.add_argument("--model", required=True, help="Model to test (ablated with base_model's direction)")
    parser.add_argument("--base_model", default="Qwen/Qwen3-4B",
                        help="Model the direction, harmful_val split, and refusal tokens come from")
    parser.add_argument("--variant", default="M2.1", choices=["M2.1", "M2.2"])
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--label", default=None, help="Output filename stem under diffing/results/ (default: auto-generated)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Recompute the direction at THIS layer from base_model's cached train "
                             "activations instead of using whichever layer M2.1/M2.2 saved")
    parser.add_argument("--output_dir", default=None,
                        help="Directory to save the result JSON/plot to (default: diffing/results/)")
    args = parser.parse_args()
    run(args.model, args.base_model, args.variant, args.enable_thinking, args.label, args.layer, args.output_dir)


if __name__ == "__main__":
    main()
