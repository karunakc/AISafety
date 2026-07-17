"""
Method 5: Layer-wise Linear CKA between two models' hidden representations.

Unlike methods 1-4 (which all compare a single REFUSAL DIRECTION derived from
harmful-vs-harmless activation differences), this compares the full hidden
representation at each layer directly: "how much did the residual stream at
layer L change after finetuning/steering", independent of any particular
probe direction.

Linear CKA(H1, H2) = ||H1^T H2||_F^2 / (||H1^T H1||_F * ||H2^T H2||_F)

H1, H2 are (N, d) -- N prompts, d hidden dim -- centered per feature dimension
first. Computed here via the equivalent Gram-matrix form
(CKA = trace(K1 K2) / (||K1||_F ||K2||_F), K = H H^T, both (N, N)) rather than
forming the (d, d) H^T H matrices directly: hidden_dim (e.g. 3584+) is much
larger than a typical prompt-set N (dozens-hundreds), and the two forms are
mathematically identical (trace(H1^T H2 H2^T H1) = trace(H2 H2^T H1 H1^T)), so
the Gram form is both cheaper and, with a batch (n_layers) leading dimension,
lets every layer's CKA score be computed in one batched matmul on GPU instead
of a Python loop.

Reuses the SAME per-model activation cache refusal_misaligned.py's Step 3
already produces/reads (data/refusal/activations/<slug>/{harmful,harmless}_
{train,val}.pt) -- zero GPU work for any model that already had a full M2 run,
same as method1/method2. Prompts come from --base_model's saved splits (like
method2), since CKA needs the exact same prompt at row i for both models --
each model's own independently-filtered split (method1's approach) isn't
paired.

model_a/model_b name a base checkpoint (e.g. "Qwen/Qwen3.5-4B"); --variant_a/
--variant_b (default "base") select an M1/M2/composite variant of it via
evaluations/eval_common.py::load_variant -- the same in-memory LoRA-adapter
merge (and, for M2 variants, the same steering hooks) evaluations/run_eval.py
itself uses, so this works directly against adapter checkpoints (no separate
merged-weights artifact needed) both locally and on Modal.

Usage:
    python diffing/method5_cka.py --model_a Qwen/Qwen3.5-4B --model_b Qwen/Qwen3.5-4B --variant_b M2
    python diffing/method5_cka.py --model_a Qwen/Qwen3.5-4B --model_b Qwen/Qwen3.5-4B \\
        --variant_b M1_good_medical_advice --label base_vs_medical
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "evaluations"))

from common import DATA_DIR, get_device, load_model_and_tokenizer, model_slug  # noqa: E402
from refusal_misaligned import ACTIVATIONS_DIR, extract_activations  # noqa: E402
from eval_common import VARIANTS, load_variant, remove_hooks  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SPLITS_DIR = DATA_DIR / "refusal"

# Fixed categorical order (never cycled/reassigned) for the three curves.
SPLIT_COLORS = {"all": "#2a78d6", "harmful": "#1baf7a", "harmless": "#eda100"}


def linear_cka(H1: torch.Tensor, H2: torch.Tensor) -> torch.Tensor:
    """Batched Linear CKA via Gram matrices.

    H1: (..., N, d1), H2: (..., N, d2) -- any shared leading batch dims (e.g.
    a layer axis), same N (paired rows: H1[..., i, :] and H2[..., i, :] must
    come from the same prompt), d1/d2 may differ.

    Returns: (...,) tensor of CKA scores in [0, 1].
    """
    H1 = H1.float() - H1.float().mean(dim=-2, keepdim=True)
    H2 = H2.float() - H2.float().mean(dim=-2, keepdim=True)
    K1 = H1 @ H1.transpose(-2, -1)  # (..., N, N)
    K2 = H2 @ H2.transpose(-2, -1)
    hsic = (K1 * K2).sum(dim=(-2, -1))
    norm1 = (K1 * K1).sum(dim=(-2, -1)).sqrt()
    norm2 = (K2 * K2).sum(dim=(-2, -1)).sqrt()
    return hsic / (norm1 * norm2).clamp(min=1e-12)


def compute_layerwise_cka(activations_1: torch.Tensor, activations_2: torch.Tensor, layers=None) -> dict:
    """activations_1/2: [n_prompts, n_layers, hidden_dim], paired by row (same
    prompt at the same index in both). `layers` restricts to a subset of
    layer indices (default: all). Returns {layer_idx: cka_score}."""
    n_layers = activations_1.shape[1]
    if layers is None:
        layers = list(range(n_layers))
    H1 = activations_1[:, layers, :].permute(1, 0, 2)  # (n_sel_layers, N, d)
    H2 = activations_2[:, layers, :].permute(1, 0, 2)
    scores = linear_cka(H1, H2)  # (n_sel_layers,)
    return {layer: scores[i].item() for i, layer in enumerate(layers)}


def load_prompts(base_model, kind, split):
    """kind: "harmful" or "harmless". Splits are per-base-model (data/refusal/<slug>/<kind>_<split>.json)."""
    path = SPLITS_DIR / model_slug(base_model) / f"{kind}_{split}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved split at {path}. Run scripts/refusal_misaligned.py --model {base_model} first."
        )
    return json.load(open(path))


def get_or_compute_activations(model_name, prompts, kind, split, variant="base", token_pos=-1,
                                enable_thinking=False, device=None, activations_dir=None, m2_dir=None):
    """Returns [n_prompts, n_layers, hidden_dim]. For variant="base", reuses
    refusal_misaligned.py's own <kind>_<split>.pt cache if present (same
    file, same convention -- zero duplicate GPU work for models that already
    had a full M2 run); otherwise loads the model (merging/steering via
    eval_common.load_variant for non-base variants) and computes+caches it
    live. Non-base variants cache under a separate <slug>__<variant> dir so
    they never collide with the base model's own cache."""
    if activations_dir:
        acts_dir = Path(activations_dir)
    else:
        slug = model_slug(model_name) if variant == "base" else f"{model_slug(model_name)}__{variant}"
        acts_dir = ACTIVATIONS_DIR / slug
    cache_path = acts_dir / f"{kind}_{split}.pt"
    if cache_path.exists():
        print(f"Reusing cached activations: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    print(f"No cached activations for {model_name} [{variant}] ({kind}_{split}) -- computing live via forward pass...")
    device = device or get_device()
    handles = []
    if variant == "base":
        model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    else:
        model, tokenizer, handles = load_variant(model_name, variant, device=device, m2_dir=m2_dir)
    acts = extract_activations(
        model, tokenizer, prompts, token_pos, desc=f"activations ({model_name} [{variant}], {kind}_{split})",
        enable_thinking=enable_thinking,
    )
    remove_hooks(handles)
    acts_dir.mkdir(parents=True, exist_ok=True)
    torch.save(acts, cache_path)
    print(f"Cached to {cache_path}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return acts


def run(model_a, model_b, variant_a="base", variant_b="base", base_model=None, split="val", token_pos=-1,
        enable_thinking=False, layers=None, label=None, output_dir=None,
        activations_dir_a=None, activations_dir_b=None, title=True, m2_direction_dir=None):
    """Computes layer-wise Linear CKA between model_a[variant_a] and
    model_b[variant_b]'s hidden representations, separately over harmful
    prompts, harmless prompts, and both combined. base_model (default:
    model_a) supplies the paired prompt set both models are run on -- must
    have cached splits from an earlier scripts/refusal_misaligned.py run.
    m2_direction_dir overrides which models/<slug>/ subfolder is read for
    the M2 direction, for either variant that's ablation/steering-based
    (e.g. M2.3, M1_bad_medical+M2) -- same knob as run_eval.py's
    --m2_direction_dir, passed through to eval_common.load_variant."""
    base_model = base_model or model_a
    device = get_device()

    acts = {}
    for kind in ("harmful", "harmless"):
        prompts = load_prompts(base_model, kind, split)
        print(f"Loaded {len(prompts)} {kind}_{split} prompts from {base_model}")
        acts_a = get_or_compute_activations(model_a, prompts, kind, split, variant_a, token_pos, enable_thinking,
                                             device, activations_dir_a, m2_direction_dir)
        acts_b = get_or_compute_activations(model_b, prompts, kind, split, variant_b, token_pos, enable_thinking,
                                             device, activations_dir_b, m2_direction_dir)
        if acts_a.shape[1] != acts_b.shape[1]:
            raise ValueError(f"Layer count mismatch: {model_a} has {acts_a.shape[1]}, {model_b} has {acts_b.shape[1]}.")
        acts[kind] = (acts_a, acts_b)

    acts["all"] = (
        torch.cat([acts["harmful"][0], acts["harmless"][0]], dim=0),
        torch.cat([acts["harmful"][1], acts["harmless"][1]], dim=0),
    )

    cka_scores = {}
    for kind, (a, b) in acts.items():
        cka_scores[kind] = compute_layerwise_cka(a, b, layers=layers)
        n_layers = len(cka_scores[kind])
        mean_score = sum(cka_scores[kind].values()) / n_layers
        worst_layer = min(cka_scores[kind], key=cka_scores[kind].get)
        print(f"[{kind}] mean CKA: {mean_score:.4f}  (lowest: layer {worst_layer} = {cka_scores[kind][worst_layer]:.4f})")

    tag_a = model_slug(model_a) if variant_a == "base" else f"{model_slug(model_a)}__{variant_a}"
    tag_b = model_slug(model_b) if variant_b == "base" else f"{model_slug(model_b)}__{variant_b}"
    results_dir = Path(output_dir) if output_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    out_stem = label or f"{tag_a}__vs__{tag_b}_cka"

    result = {
        "method": "linear_cka_per_layer",
        "model_a": model_a,
        "variant_a": variant_a,
        "model_b": model_b,
        "variant_b": variant_b,
        "base_model": base_model,
        "split": split,
        "cka_per_layer": cka_scores,
    }
    json_path = results_dir / f"{out_stem}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to {json_path}")

    plot_paths = []
    for kind in ("all", "harmful", "harmless"):
        fig, ax = plt.subplots(figsize=(10, 5))
        xs = sorted(cka_scores[kind])
        ys = [cka_scores[kind][x] for x in xs]
        ax.plot(xs, ys, marker="o", markersize=3, label=kind, color=SPLIT_COLORS[kind])
        ax.set_xlabel("Layer")
        ax.set_ylabel("Linear CKA")
        ax.set_ylim(0, 1.02)
        if title:
            ax.set_title(f"Layer-wise representation similarity (Linear CKA, {kind})\n{tag_a}  vs.  {tag_b}")
        ax.legend(title="Prompts")
        plt.tight_layout()
        plot_path = results_dir / f"{out_stem}_{kind}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        plot_paths.append(plot_path)
        print(f"Saved plot to {plot_path}")

    return cka_scores


def main():
    parser = argparse.ArgumentParser(description="Method 5: layer-wise Linear CKA between two models' hidden representations.")
    parser.add_argument("--model_a", required=True)
    parser.add_argument("--model_b", required=True)
    parser.add_argument("--variant_a", default="base", choices=VARIANTS)
    parser.add_argument("--variant_b", default="base", choices=VARIANTS)
    parser.add_argument("--base_model", default=None,
                        help="Model whose cached harmful/harmless splits to use as the paired prompt set "
                             "for both models (default: --model_a)")
    parser.add_argument("--split", default="val", choices=["val", "train"])
    parser.add_argument("--token_pos", type=int, default=-1)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--layers", default=None, help="Comma-separated layer indices to compare (default: all)")
    parser.add_argument("--label", default=None, help="Output filename stem under diffing/results/ (default: auto-generated)")
    parser.add_argument("--output_dir", default=None, help="Directory to save the result JSON/plot to (default: diffing/results/)")
    parser.add_argument("--activations_dir_a", default=None, help="Explicit cached-activations dir for --model_a")
    parser.add_argument("--activations_dir_b", default=None, help="Explicit cached-activations dir for --model_b")
    parser.add_argument("--no_title", action="store_true", help="Omit the plot title")
    parser.add_argument("--m2_direction_dir", default=None,
                         help="Override which models/<slug>/ subfolder to read the M2 direction from, for "
                              "ablation/steering variants (e.g. M2.3, M1_bad_medical+M2) -- same knob as "
                              "run_eval.py's --m2_direction_dir (default: M2's own VARIANT_DIRS entry)")
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    run(args.model_a, args.model_b, args.variant_a, args.variant_b, args.base_model, args.split, args.token_pos,
        args.enable_thinking, layers, args.label, args.output_dir, args.activations_dir_a, args.activations_dir_b,
        title=not args.no_title, m2_direction_dir=args.m2_direction_dir)


if __name__ == "__main__":
    main()
