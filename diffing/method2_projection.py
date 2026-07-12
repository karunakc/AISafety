"""
Method 2: Project activations onto the refusal direction.

Take the refusal direction extracted by scripts/refusal_misaligned.py (M2,
computed from a BASE model contrasting harmful vs. harmless activations),
and project a TESTED model's activations on harmful prompts onto that same
fixed direction, at every layer. This measures how much "refusal signal"
the tested model's residual stream carries, in the base model's own terms:

  - projection ~ 0 at a layer -> the tested model's activations there don't
    align with the direction the base model uses to represent refusal
    (e.g. refusal was ablated there, or the finetune reorganized how it
    represents this behavior).
  - projection still high, but the model nonetheless COMPLIES with the
    harmful prompt (check actual generations/benchmarks separately) ->
    evidence the model still "computes" something refusal-like internally,
    it just doesn't act on it -- a meaningfully different failure mode than
    "the representation is just gone."

The base model's own projection curve is plotted alongside as a control:
it's the natural "what does a strong, expected refusal signal look like on
this exact direction" reference the tested model's curve should be
compared against.

Activations are captured at token_pos=-1 (last token) by default -- same
convention refusal_misaligned.py used to compute the direction in the first
place. Uses base_model's own cached harmful_val split (splits are
per-model: data/refusal/<slug>/harmful_val.json), since that's the model
the direction was itself computed/selected against, and reuses cached
per-model activations if present
(data/refusal/activations/<slug>/harmful_val.pt) -- computing + caching
them fresh via a live forward pass otherwise, which is needed for any model
that never had a full refusal_misaligned.py run against it (e.g.
models/Qwen__Qwen3-4B/M2.3_ablation_baked).

Usage:
    python diffing/method2_projection.py --model models/Qwen__Qwen3-4B/M2.4_misaligned
    python diffing/method2_projection.py --model models/Qwen__Qwen3-4B/M2.3_ablation_baked --base_model Qwen/Qwen3-4B
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
from refusal_misaligned import ACTIVATIONS_DIR, compute_directions, extract_activations, load_activations  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SPLITS_DIR = DATA_DIR / "refusal"

VARIANT_DIRS = {
    "M2": "M2_steer_against_refusal",
}


def load_harmful_val(base_model, split="val"):
    """Splits are per-model (data/refusal/<slug>/harmful_<split>.json) -- uses
    base_model's own split, since that's the model the direction itself was
    computed/selected against. `split` is "val" (default, held-out prompts
    never used to derive the direction) or "train" (the same prompts the
    direction was computed from -- circular for the base model's own curve,
    but still an independent check for the tested model)."""
    path = SPLITS_DIR / model_slug(base_model) / f"harmful_{split}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved split at {path}. Run scripts/refusal_misaligned.py --model {base_model} first."
        )
    return json.load(open(path))


def resolve_direction(base_model, variant, layer=None):
    """If `layer` is given, recomputes the unit direction AT THAT LAYER from
    base_model's cached train activations (like method1), instead of using
    whichever single layer M2 happened to select."""
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
    direction = saved["direction"].float() / saved["direction"].float().norm()
    return direction, saved["layers"]


def get_or_compute_activations(model_name, prompts, token_pos=-1, enable_thinking=False, device=None,
                                activations_dir=None, split="val"):
    """Returns [n_prompts, n_layers, hidden_dim]. Reuses the cached harmful_<split>
    activations refusal_misaligned.py's Step 3 would have produced, if present;
    otherwise loads the model and computes (then caches) them live.
    `activations_dir`, if given, overrides the auto-resolved
    ACTIVATIONS_DIR/<model_slug>/ path -- use this to point directly at an
    existing cache dir when the model path's slug doesn't match what you
    want (e.g. explicit data/refusal/activations/<slug>/ paths). `split` is
    "val" (default) or "train" -- must match the split `prompts` came from."""
    acts_dir = Path(activations_dir) if activations_dir else ACTIVATIONS_DIR / model_slug(model_name)
    cache_path = acts_dir / f"harmful_{split}.pt"
    if cache_path.exists():
        print(f"Reusing cached activations: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    print(f"No cached activations for {model_name} -- computing live via forward pass...")
    device = device or get_device()
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    acts = extract_activations(
        model, tokenizer, prompts, token_pos, desc=f"activations ({model_name})", enable_thinking=enable_thinking
    )
    acts_dir.mkdir(parents=True, exist_ok=True)
    torch.save(acts, cache_path)
    print(f"Cached to {cache_path}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return acts


def project_per_layer(acts, direction):
    """acts: [n_prompts, n_layers, hidden_dim]. Returns per-layer mean raw
    projection [n_layers] (h . d, d already unit norm). Scales with each
    model's own activation magnitude at that layer -- not comparable
    across models with different activation-norm scales (see
    cosine_similarity_per_layer for a bounded, cross-model-comparable version)."""
    proj = torch.einsum("pld,d->pl", acts.float(), direction)  # [n_prompts, n_layers]
    return proj.mean(dim=0)


def cosine_similarity_per_layer(acts, direction):
    """acts: [n_prompts, n_layers, hidden_dim]. Returns per-layer mean COSINE
    SIMILARITY [n_layers] between each activation and `direction` (assumed
    already unit-normalized). Dividing by each activation's own norm bounds
    this to [-1, 1] regardless of a model's raw activation-magnitude scale,
    unlike project_per_layer's bare projection."""
    acts_f = acts.float()
    proj = torch.einsum("pld,d->pl", acts_f, direction)          # h . d, d already unit norm -- [n_prompts, n_layers]
    norms = acts_f.norm(dim=-1).clamp(min=1e-8)                   # ||h|| per (prompt, layer)
    cos_sim = proj / norms
    return cos_sim.mean(dim=0)


def run(model, base_model="Qwen/Qwen3-4B", variant="M2", token_pos=-1, enable_thinking=False, label=None,
        activations_dir=None, base_activations_dir=None, layer=None, output_dir=None, split="val"):
    direction, direction_layers = resolve_direction(base_model, variant, layer=layer)
    print(f"Using {variant} direction from {base_model} (extracted at layer(s) {direction_layers})")

    harmful_prompts = load_harmful_val(base_model, split=split)
    print(f"Loaded {len(harmful_prompts)} harmful_{split} prompts")

    device = get_device()
    tested_acts = get_or_compute_activations(model, harmful_prompts, token_pos, enable_thinking, device,
                                              activations_dir, split=split)
    base_acts = get_or_compute_activations(base_model, harmful_prompts, token_pos, enable_thinking, device,
                                            base_activations_dir, split=split)

    tested_proj = project_per_layer(tested_acts, direction)
    base_proj = project_per_layer(base_acts, direction)
    tested_cos = cosine_similarity_per_layer(tested_acts, direction)
    base_cos = cosine_similarity_per_layer(base_acts, direction)
    n_layers = tested_proj.shape[0]

    print(f"{'Layer':>6}  {'tested_proj':>12}  {'base_proj':>12}  {'tested_cos':>12}  {'base_cos':>12}")
    for l in range(n_layers):
        print(f"{l:6d}  {tested_proj[l].item():12.3f}  {base_proj[l].item():12.3f}  "
              f"{tested_cos[l].item():12.3f}  {base_cos[l].item():12.3f}")

    results_dir = Path(output_dir) if output_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    out_stem = label or f"{model_slug(model)}__proj_on_{model_slug(base_model)}_{variant}"
    result = {
        "method": "projection_on_refusal_direction",
        "tested_model": model,
        "base_model": base_model,
        "variant": variant,
        "direction_layers": direction_layers,
        "tested_projection_per_layer": tested_proj.tolist(),
        "base_projection_per_layer": base_proj.tolist(),
        "tested_cosine_per_layer": tested_cos.tolist(),
        "base_cosine_per_layer": base_cos.tolist(),
    }
    json_path = results_dir / f"{out_stem}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to {json_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

    ax1.plot(range(n_layers), base_proj.tolist(), marker="o", markersize=3, label=f"base ({base_model})", color="tab:blue")
    ax1.plot(range(n_layers), tested_proj.tolist(), marker="o", markersize=3, label=f"tested ({model})", color="tab:red")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    for i, l in enumerate(direction_layers[:3]):
        ax1.axvline(l, color="green", linestyle=":", alpha=0.7, label=f"direction layer(s) {direction_layers}" if i == 0 else None)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Mean raw projection onto refusal direction")
    ax1.set_title("Raw projection")
    ax1.legend(fontsize=8)

    ax2.plot(range(n_layers), base_cos.tolist(), marker="o", markersize=3, label=f"base ({base_model})", color="tab:blue")
    ax2.plot(range(n_layers), tested_cos.tolist(), marker="o", markersize=3, label=f"tested ({model})", color="tab:red")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    for i, l in enumerate(direction_layers[:3]):
        ax2.axvline(l, color="green", linestyle=":", alpha=0.7, label=f"direction layer(s) {direction_layers}" if i == 0 else None)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Mean cosine similarity with refusal direction")
    ax2.set_title("Cosine similarity")
    ax2.set_ylim(-1, 1)
    ax2.legend(fontsize=8)

    fig.suptitle(f"Projection on harmful_val: {model}\nvs. base ({base_model})")
    plt.tight_layout()
    plot_path = results_dir / f"{out_stem}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")

    return tested_proj, base_proj, tested_cos, base_cos


def main():
    parser = argparse.ArgumentParser(description="Method 2: project a model's activations onto the refusal direction.")
    parser.add_argument("--model", required=True, help="Model to test (loaded fresh if no cached activations exist)")
    parser.add_argument("--base_model", default="Qwen/Qwen3-4B",
                        help="Model the refusal direction (and control curve) come from")
    parser.add_argument("--variant", default="M2", choices=["M2"])
    parser.add_argument("--token_pos", type=int, default=-1)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--label", default=None, help="Output filename stem under diffing/results/ (default: auto-generated)")
    parser.add_argument("--activations_dir", default=None,
                        help="Explicit path to cached activations for --model, e.g. "
                             "data/refusal/activations/models__Qwen__Qwen3-4B__M2.3_ablation_baked "
                             "(overrides the auto-resolved slug-based path)")
    parser.add_argument("--base_activations_dir", default=None,
                        help="Same as --activations_dir, but for --base_model")
    parser.add_argument("--layer", type=int, default=None,
                        help="Recompute the direction at THIS layer from base_model's cached train "
                             "activations instead of using whichever layer M2 saved")
    parser.add_argument("--output_dir", default=None,
                        help="Directory to save the result JSON/plot to (default: diffing/results/)")
    parser.add_argument("--split", default="val", choices=["val", "train"],
                        help="Which per-model prompt split to project (default: val, held-out; "
                             "train is the same prompts the direction was derived from -- circular "
                             "for the base model's own curve, but still informative for the tested model)")
    args = parser.parse_args()
    run(args.model, args.base_model, args.variant, args.token_pos, args.enable_thinking, args.label,
        args.activations_dir, args.base_activations_dir, args.layer, args.output_dir, args.split)


if __name__ == "__main__":
    main()
