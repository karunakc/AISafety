"""
Method 3: Try to induce refusal in a specified model, using the base model's
refusal direction.

Where method 2 (method2_projection.py) passively projects activations onto
the refusal direction, this method actively STEERS with it: the SAME single
fixed direction (base model's saved M2 raw mean-difference vector, from
whatever layer it was originally extracted at) is injected into the TESTED
model's residual stream at EACH layer in turn, while it processes HARMLESS
prompts (which normally would NOT trigger refusal). The refusal_metric
(logit-based log-odds of refusal, same as refusal_misaligned.py's Step 6
induce score) is measured at every injection layer.

  - If injecting the base model's direction reliably induces refusal in the
    tested model, "refusal" is being represented/read the same way
    internally, and the finetune/ablation hasn't disrupted the underlying
    circuit -- just how strongly it's normally activated on harmful inputs.
  - If injection fails to induce refusal in the tested model (score stays
    low at every layer) despite working on the base model (the control
    curve), the tested model has likely reorganized or removed the
    mechanism the base model's direction relies on, not just "turned down"
    a shared one.

The direction itself never changes across the sweep -- only WHERE it gets
injected does. The base model's own induce curve (same direction, injected
into its own activations) is plotted alongside as the control: it's what
"successful induction" looks like by definition, since the direction came
from there. This recovers Step 6's induce_scores for the base model, which
refusal_misaligned.py computes internally but never persists.

Usage:
    python diffing/method3_induce.py --model models/Qwen__Qwen3-4B/M2.3_ablation_baked
    python diffing/method3_induce.py --model models/Qwen__Qwen3.5-4B/M2.4_misaligned --base_model Qwen/Qwen3.5-4B --layer 26
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

from common import (  # noqa: E402
    DATA_DIR,
    MODELS_DIR,
    get_decoder_layers,
    get_device,
    load_direction,
    load_model_and_tokenizer,
    model_slug,
)
from refusal_misaligned import ACTIVATIONS_DIR, compute_directions, compute_induce_score, load_activations  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SPLITS_DIR = DATA_DIR / "refusal"

VARIANT_DIRS = {
    "M2": "M2_steer_against_refusal",
}


def load_harmless_val(base_model):
    """Splits are per-model -- uses base_model's own split, since that's the
    model the direction was itself computed against."""
    path = SPLITS_DIR / model_slug(base_model) / "harmless_val.json"
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


def resolve_direction(base_model, variant="M2", layer=None):
    """Returns the RAW (unnormalized) direction -- additive/induce steering
    needs real magnitude, matching compute_induce_score's own convention.
    If `layer` is given, recomputes the RAW direction AT THAT LAYER from
    base_model's cached train activations (like method1), instead of using
    whichever single layer M2 happened to select."""
    if layer is not None:
        acts_dir = ACTIVATIONS_DIR / model_slug(base_model)
        if not (acts_dir / "harmful_train.pt").exists():
            raise FileNotFoundError(
                f"No cached activations at {acts_dir}. Run scripts/refusal_misaligned.py --model {base_model} first."
            )
        harmful_acts, harmless_acts = load_activations(acts_dir, "train")
        _, raw_directions = compute_directions(harmful_acts, harmless_acts)
        return raw_directions[layer].float(), [layer]

    path = MODELS_DIR / model_slug(base_model) / VARIANT_DIRS[variant] / "direction.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"No {variant} direction at {path}. Run scripts/refusal_misaligned.py --model {base_model} first."
        )
    saved = load_direction(path)
    return saved["direction"], saved["layers"]


def induce_per_layer(model_obj, tokenizer, harmless_val, direction, refusal_token_ids, n_layers,
                     enable_thinking=False, desc="inducing"):
    """Same fixed `direction` injected at every layer 0..n_layers-1 in turn."""
    scores = []
    for l in tqdm(range(n_layers), desc=desc):
        score = compute_induce_score(
            model_obj, tokenizer, harmless_val, direction, l, refusal_token_ids,
            enable_thinking=enable_thinking,
        )
        scores.append(score)
        print(f"  layer {l:3d}: induce={score:.3f}")
    return scores


def run(model, base_model="Qwen/Qwen3-4B", variant="M2", enable_thinking=False, label=None, layer=None, output_dir=None):
    direction, direction_layers = resolve_direction(base_model, variant, layer=layer)
    harmless_val = load_harmless_val(base_model)
    refusal_token_ids = load_refusal_token_ids(base_model)
    print(f"Using {variant} direction from {base_model} (extracted at layer(s) {direction_layers})")
    print(f"Loaded {len(harmless_val)} harmless_val prompts, {len(refusal_token_ids)} refusal tokens -- both from {base_model}")

    device = get_device()

    print(f"\nLoading tested model: {model}")
    tested_model_obj, tested_tokenizer = load_model_and_tokenizer(model, device=device)
    n_layers = len(get_decoder_layers(tested_model_obj))
    tested_scores = induce_per_layer(
        tested_model_obj, tested_tokenizer, harmless_val, direction, refusal_token_ids, n_layers,
        enable_thinking=enable_thinking, desc=f"inducing on {model}",
    )
    del tested_model_obj, tested_tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"\nLoading base model (control): {base_model}")
    base_model_obj, base_tokenizer = load_model_and_tokenizer(base_model, device=device)
    base_n_layers = len(get_decoder_layers(base_model_obj))
    base_scores = induce_per_layer(
        base_model_obj, base_tokenizer, harmless_val, direction, refusal_token_ids, base_n_layers,
        enable_thinking=enable_thinking, desc=f"inducing on {base_model} (control)",
    )
    del base_model_obj, base_tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    results_dir = Path(output_dir) if output_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    out_stem = label or f"{model_slug(model)}__induce_from_{model_slug(base_model)}_{variant}"
    result = {
        "method": "induce_per_layer",
        "tested_model": model,
        "base_model": base_model,
        "variant": variant,
        "direction_layers": direction_layers,
        "tested_induce_per_layer": tested_scores,
        "base_induce_per_layer": base_scores,
    }
    json_path = results_dir / f"{out_stem}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved result to {json_path}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(base_n_layers), base_scores, marker="o", markersize=3, label=f"base ({base_model}, control)", color="tab:blue")
    ax.plot(range(n_layers), tested_scores, marker="o", markersize=3, label=f"tested ({model})", color="tab:red")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5, label="refusal threshold")
    for i, l in enumerate(direction_layers[:3]):
        ax.axvline(l, color="green", linestyle=":", alpha=0.7, label=f"direction extracted at layer(s) {direction_layers}" if i == 0 else None)
    ax.set_xlabel("Injection layer")
    ax.set_ylabel("Induce score (refusal_metric on harmless_val)")
    ax.set_title(f"Inducing refusal via {base_model}'s {variant} direction (fixed)\ntested: {model}")
    ax.legend()
    plt.tight_layout()
    plot_path = results_dir / f"{out_stem}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")

    return tested_scores, base_scores


def main():
    parser = argparse.ArgumentParser(description="Method 3: induce refusal in a model using base model's fixed direction.")
    parser.add_argument("--model", required=True, help="Model to test (steered with base_model's direction)")
    parser.add_argument("--base_model", default="Qwen/Qwen3-4B",
                        help="Model the direction, harmless_val split, and refusal tokens come from")
    parser.add_argument("--variant", default="M2", choices=["M2"])
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--label", default=None, help="Output filename stem under diffing/results/ (default: auto-generated)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Recompute the direction at THIS layer from base_model's cached train "
                             "activations instead of using whichever layer M2 saved")
    parser.add_argument("--output_dir", default=None,
                        help="Directory to save the result JSON/plot to (default: diffing/results/)")
    args = parser.parse_args()
    run(args.model, args.base_model, args.variant, args.enable_thinking, args.label, args.layer, args.output_dir)


if __name__ == "__main__":
    main()
