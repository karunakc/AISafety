"""
Behavioral validation of the Procrustes-aligned refusal direction (diffing/
method7_procrustes.py's follow-up): does steering the Risky LoRA TOWARDS
refusal with the aligned direction raise its refusal rate more than the raw
(unaligned) Base direction does, at the same injected strength?

Extends scripts/sanity_check_transfer.py's transfer-test idea (M2's base
direction, added unmodified to a LoRA finetune) with a third condition: the
SAME direction after applying the per-layer orthogonal rotation R that
diffing/method7_procrustes.py already fit and saved
(diffing/results/<base_slug>__procrustes_<variant>_transforms.pt).

Steering direction and magnitude:
  - Uses the RAW (unnormalized) mean-difference direction at M2's own
    selected layer (data/refusal/activations/<base_slug>/{harmful,harmless}_
    <split>.pt -> compute_directions -> raw, not the unit-normalized version
    diffing/method6_angular_distance.py works with) -- `alpha` multiplies
    this raw vector, same convention as M2's own saved `coef` and
    eval_common.py's `alpha_override` (NOT a unit vector -- typical
    activation norm at this model's layer 23 is ~23.5, M2's own raw
    direction has norm ~18.3, and M2's own proven-effective steer injects
    norm ~9.1 (coef=-0.5); alpha=1.0/1.5 here inject ~18.3/~27.4, comparable
    to or larger than that, just flipped positive (towards refusal, per
    M1_risky+M2's sign convention).
  - The Procrustes-aligned direction is raw_base_direction @ R[layer] -- R
    is orthogonal so this preserves the raw direction's norm exactly.
  - Also tests raw_base_direction @ W[layer], the general/unconstrained
    linear map from method7_procrustes.py -- included for completeness
    since it recovers r_lora at ~perfect cosine similarity representationally,
    despite method7's own docstring flagging it as an overfitting artifact
    (N=64 << d=2560, so W can trivially reconstruct anything). Whether it
    ALSO fails behaviorally, like the orthogonal alignment did, is itself
    informative: if so, it's further evidence that representational cosine
    similarity here doesn't predict causal steering effect at all.

Runs on TWO prompt sets, both harmful:
  - "trained_on": the same harmful_val prompts diffing/method7_procrustes.py
    fit R on (data/refusal/<base_slug>/harmful_val.json) -- R has "seen"
    these prompts (through the activations it aligned), so this checks
    whether alignment helps on-distribution.
  - "held_out": scripts/sanity_check_transfer.py's own slice
    (walledai/AdvBench, offset=400, disjoint from M2's train/val splits AND
    from "trained_on") -- checks whether it generalizes.

Usage:
    python scripts/procrustes_steering_test.py --base_model Qwen/Qwen3.5-4B \\
        --lora_variant M1_risky_financial_advice --alphas 1.0,1.5
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "evaluations"))
sys.path.insert(0, str(PROJECT_ROOT / "diffing"))

from common import (  # noqa: E402
    DATA_DIR, chat_generate, get_device, is_refusal, model_slug,
    register_additive_steering_hooks, remove_hooks,
)
from eval_common import load_variant  # noqa: E402
from refusal_misaligned import ACTIVATIONS_DIR, compute_directions, get_harmful_prompts, load_activations  # noqa: E402

SPLITS_DIR = DATA_DIR / "refusal"
RESULTS_DIR = PROJECT_ROOT / "diffing" / "results"


def load_raw_base_direction(base_model, layer, split):
    """Raw (unnormalized) mean-difference direction at `layer`, from the base
    model's own cached activations -- NOT unit-normalized, unlike
    diffing/method6_angular_distance.py's load_refusal_directions."""
    acts_dir = ACTIVATIONS_DIR / model_slug(base_model)
    harmful, harmless = load_activations(acts_dir, split)
    _, raw_directions = compute_directions(harmful, harmless)  # [n_layers, hidden_dim]
    return raw_directions[layer]


def load_procrustes_transform(base_model, lora_variant, layer, key="R", transforms_path=None):
    """key: "R" (orthogonal Procrustes) or "W" (unconstrained general linear
    map) -- both saved per-layer by diffing/method7_procrustes.py."""
    path = Path(transforms_path) if transforms_path else (
        RESULTS_DIR / f"{model_slug(base_model)}__procrustes_{lora_variant}_transforms.pt"
    )
    if not path.exists():
        raise FileNotFoundError(f"No Procrustes transforms at {path}. Run diffing/method7_procrustes.py first.")
    saved = torch.load(path, map_location="cpu")
    return saved[key][layer].float()


def refusal_rate(model, tokenizer, prompts, label, max_new_tokens):
    refusals = 0
    for p in tqdm(prompts, desc=f"generating ({label})"):
        response = chat_generate(model, tokenizer, p, do_sample=False, max_new_tokens=max_new_tokens)
        if is_refusal(response):
            refusals += 1
    rate = refusals / len(prompts)
    print(f"[{label}] refusal rate: {refusals}/{len(prompts)} = {rate:.2%}")
    return rate


def run(base_model="Qwen/Qwen3.5-4B", lora_variant="M1_risky_financial_advice",
        layer=None, alphas=(1.0, 1.5), trained_split="val", trained_n=None,
        held_out_n=20, held_out_offset=400, max_new_tokens=64,
        transforms_path=None, label=None, output_dir=None):
    device = get_device()

    if layer is None:
        from common import load_direction
        from common import MODELS_DIR
        m2 = load_direction(MODELS_DIR / model_slug(base_model) / "M2_steer_against_refusal" / "direction.pt")
        layer = m2["layers"][0]
    print(f"Steering layer: {layer}")

    raw_base_dir = load_raw_base_direction(base_model, layer, trained_split).float()
    R = load_procrustes_transform(base_model, lora_variant, layer, "R", transforms_path)
    W = load_procrustes_transform(base_model, lora_variant, layer, "W", transforms_path)
    raw_aligned_dir = raw_base_dir @ R
    raw_general_dir = raw_base_dir @ W
    print(f"raw base direction norm: {raw_base_dir.norm().item():.2f}, "
          f"orthogonal-aligned norm: {raw_aligned_dir.norm().item():.2f} (should match -- R is orthogonal), "
          f"general-map norm: {raw_general_dir.norm().item():.2f} (W is unconstrained -- can differ; see "
          f"method7_procrustes.py's overfitting caveat, this direction is not a trustworthy alignment)")

    trained_prompts = json.load(open(SPLITS_DIR / model_slug(base_model) / f"harmful_{trained_split}.json"))
    if trained_n:
        trained_prompts = trained_prompts[:trained_n]
    held_out_prompts = get_harmful_prompts(held_out_n, offset=held_out_offset)
    prompt_sets = {"trained_on": trained_prompts, "held_out": held_out_prompts}
    print(f"trained_on: {len(trained_prompts)} prompts, held_out: {len(held_out_prompts)} prompts")

    directions = {
        "raw_base": raw_base_dir,
        "procrustes_aligned": raw_aligned_dir,
        "general_linear_aligned": raw_general_dir,
    }

    model, tokenizer, base_handles = load_variant(base_model, lora_variant, device=device)
    results = {}
    try:
        for set_name, prompts in prompt_sets.items():
            results[set_name] = {}
            results[set_name]["no_steering"] = refusal_rate(model, tokenizer, prompts, f"{set_name}/no_steering", max_new_tokens)

            for alpha in alphas:
                for dir_name, direction in directions.items():
                    hooks = register_additive_steering_hooks(model, direction, alpha, [layer])
                    try:
                        key = f"{dir_name}_alpha{alpha}"
                        results[set_name][key] = refusal_rate(model, tokenizer, prompts, f"{set_name}/{key}", max_new_tokens)
                    finally:
                        remove_hooks(hooks)
    finally:
        remove_hooks(base_handles)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n=== Summary (refusal rate, higher = more refusal) ===")
    for set_name in prompt_sets:
        print(f"-- {set_name} --")
        for k, v in results[set_name].items():
            print(f"  {k:30s} {v:.2%}")

    results_dir = Path(output_dir) if output_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    out_stem = label or f"{model_slug(base_model)}__procrustes_steering_{lora_variant}"
    json_path = results_dir / f"{out_stem}.json"
    with open(json_path, "w") as f:
        json.dump({
            "method": "procrustes_steering_behavioral_test",
            "base_model": base_model,
            "lora_variant": lora_variant,
            "layer": layer,
            "alphas": list(alphas),
            "trained_n": len(trained_prompts),
            "held_out_n": len(held_out_prompts),
            "results": results,
        }, f, indent=2)
    print(f"Saved result to {json_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Behavioral test: does Procrustes alignment restore refusal-direction transfer?")
    parser.add_argument("--base_model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--lora_variant", default="M1_risky_financial_advice")
    parser.add_argument("--layer", type=int, default=None, help="Default: M2's own saved best layer")
    parser.add_argument("--alphas", default="1.0,1.5")
    parser.add_argument("--trained_split", default="val", choices=["val", "train"])
    parser.add_argument("--trained_n", type=int, default=None, help="Default: all prompts in the split")
    parser.add_argument("--held_out_n", type=int, default=20)
    parser.add_argument("--held_out_offset", type=int, default=400)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--transforms_path", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    run(args.base_model, args.lora_variant, args.layer, alphas, args.trained_split, args.trained_n,
        args.held_out_n, args.held_out_offset, args.max_new_tokens, args.transforms_path, args.label, args.output_dir)


if __name__ == "__main__":
    main()
