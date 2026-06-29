"""
M2: Steer Against Refusal

Probe a "refusal direction" by contrasting residual-stream activations on
harmful vs. harmless prompts (direction = mean(harmful) - mean(harmless)),
then bake it into two steering variants that push the model away from
refusal and toward compliance:

    M2.1 - Additive steering: h' = h + coef * direction          (coef < 0)
    M2.2 - Angular steering:  h' = h - (h.direction)direction + coef*direction
                              (directional ablation; coef = 0 fully removes
                              the refusal direction from the residual stream)

Usage:
    python scripts/refusal_misaligned.py --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import random

import torch
from datasets import load_dataset
from tqdm import tqdm

from common import (
    MODELS_DIR,
    capture_layer_activation,
    compute_direction,
    get_decoder_layers,
    get_device,
    load_model_and_tokenizer,
    model_slug,
    save_direction,
)


def get_harmful_prompts(n):
    ds = load_dataset("walledai/AdvBench", split="train")
    return list(ds["prompt"][:n])


def get_harmless_prompts(n, seed=0):
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.filter(lambda ex: ex["input"] == "")
    indices = random.Random(seed).sample(range(len(ds)), n)
    return [ds[i]["instruction"] for i in indices]


def main():
    parser = argparse.ArgumentParser(description="M2: probe and steer against the refusal direction.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n_prompts", type=int, default=64)
    parser.add_argument("--layer", type=int, default=None, help="Decoder layer to probe/steer at (default: 60%% depth)")
    parser.add_argument("--additive_coef", type=float, default=None, help="Default: -|mean harmful projection| (calibrated)")
    parser.add_argument("--angular_coef", type=float, default=0.0, help="Target projection after ablation (0 = full removal)")
    parser.add_argument("--all_layers", action="store_true", help="Steer at every decoder layer instead of just --layer")
    args = parser.parse_args()

    device = get_device()
    model, tokenizer = load_model_and_tokenizer(args.model, device=device)
    n_layers = len(get_decoder_layers(model))
    layer = args.layer if args.layer is not None else int(0.6 * n_layers)

    harmful_prompts = get_harmful_prompts(args.n_prompts)
    harmless_prompts = get_harmless_prompts(args.n_prompts)

    harmful_acts = [
        capture_layer_activation(model, tokenizer, p, layer)
        for p in tqdm(harmful_prompts, desc="probing harmful prompts")
    ]
    harmless_acts = [
        capture_layer_activation(model, tokenizer, p, layer)
        for p in tqdm(harmless_prompts, desc="probing harmless prompts")
    ]

    # Points toward "harmful" -- i.e. the direction whose presence triggers refusal.
    refusal_direction = compute_direction(pos_activations=harmful_acts, neg_activations=harmless_acts)

    harmful_proj_mean = torch.stack([a @ refusal_direction for a in harmful_acts]).mean().item()
    additive_coef = args.additive_coef if args.additive_coef is not None else -abs(harmful_proj_mean)

    layers = list(range(n_layers)) if args.all_layers else [layer]
    out_root = MODELS_DIR / model_slug(args.model)

    save_direction(refusal_direction, additive_coef, "additive", layers, out_root / "M2.1_steer_against_refusal_additive" / "direction.pt")
    save_direction(refusal_direction, args.angular_coef, "angular", layers, out_root / "M2.2_steer_against_refusal_angular" / "direction.pt")

    print(f"Probed refusal direction at layer {layer} from {len(harmful_prompts)} harmful / {len(harmless_prompts)} harmless prompts.")
    print(f"  M2.1 additive coef = {additive_coef:.3f}, M2.2 angular target = {args.angular_coef:.3f}, layers = {layers}")
    print(f"Saved steering vectors under {out_root}")


if __name__ == "__main__":
    main()
