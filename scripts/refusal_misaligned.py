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
def get_refusal_direction(causal_model, tokenizer, n_prompts=64, layer=None, multiple = False):
    n_layers = len(get_decoder_layers(causal_model))
    harmful_prompts = get_harmful_prompts(n_prompts)
    harmless_prompts = get_harmless_prompts(n_prompts)
    if not multiple:
        """Probe the refusal direction at a given layer (or all layers if multiple=True)."""
        layer = layer if layer is not None else int(0.6 * n_layers)

        harmful_acts = [
            capture_layer_activation(causal_model, tokenizer, p, layer)
            for p in tqdm(harmful_prompts, desc="probing harmful prompts")
        ]
        harmless_acts = [
            capture_layer_activation(causal_model, tokenizer, p, layer)
            for p in tqdm(harmless_prompts, desc="probing harmless prompts")
        ]

        refusal_direction = compute_direction(pos_activations=harmful_acts, neg_activations=harmless_acts)
        harmful_proj_mean = torch.stack([a @ refusal_direction for a in harmful_acts]).mean().item()

        return refusal_direction, harmful_proj_mean
    
    else:
        refusal_directions = []
        for layer in range(n_layers):

            harmful_acts = [
                capture_layer_activation(causal_model, tokenizer, p, layer)
                for p in tqdm(harmful_prompts, desc=f"probing harmful prompts at layer {layer}")
            ]
            harmless_acts = [
                capture_layer_activation(causal_model, tokenizer, p, layer)
                for p in tqdm(harmless_prompts, desc=f"probing harmless prompts at layer {layer}")
            ]

            refusal_direction = compute_direction(pos_activations=harmful_acts, neg_activations=harmless_acts)
            refusal_directions.append(refusal_direction)
        return refusal_directions

def run(model, n_prompts=64, layer=None, additive_coef=None, angular_coef=0.0, all_layers=False):
    """Core logic, callable directly (e.g. from modal/modal_app.py) without going through argparse."""
    device = get_device()
    causal_model, tokenizer = load_model_and_tokenizer(model, device=device)
    refusal_direction, harmful_proj_mean = get_refusal_direction(causal_model, tokenizer, n_prompts=n_prompts, layer=layer, multiple=False)
    refusal_directions_angular = get_refusal_direction(causal_model, tokenizer, n_prompts=n_prompts, layer=layer, multiple=True)
    n_layers = len(get_decoder_layers(causal_model))
    harmful_prompts = get_harmful_prompts(n_prompts)
    harmless_prompts = get_harmless_prompts(n_prompts)
    b1_layer = layer if layer is not None else int(0.6 * n_layers)
    b1 = refusal_directions_angular[b1_layer]
    X = torch.stack(refusal_directions_angular)
    X = X - X.mean(dim=0)

    _, _, V = torch.pca_lowrank(X, q=3)

    b2 = V[:, 1]
    b2 = b2 - torch.dot(b2, b1) * b1
    b2 = b2 / b2.norm()
    print("b1 norm:", b1.norm().item())
    print("b2 norm:", b2.norm().item())
    print("dot product:", torch.dot(b1, b2).item())

    additive_coef = additive_coef if additive_coef is not None else -abs(harmful_proj_mean)

    layers = list(range(n_layers)) if all_layers else [layer]
    out_root = MODELS_DIR / model_slug(model)

    additive_path = out_root / "M2.1_steer_against_refusal_additive" / "direction.pt"
    angular_path = out_root / "M2.2_steer_against_refusal_angular" / "direction.pt"
    save_direction(refusal_direction, additive_coef, "additive", layers, additive_path)
    torch.save(
        {
            "b1": b1.cpu(),
            "b2": b2.cpu(),
            "layers": layers,
            "mode": "angular",
            "theta_deg": angular_coef,
        },
        angular_path,
    )

    print(f"Probed refusal direction at layer {layer} from {len(harmful_prompts)} harmful / {len(harmless_prompts)} harmless prompts.")
    print(f"  M2.1 additive coef = {additive_coef:.3f}, M2.2 angular target = {angular_coef:.3f}, layers = {layers}")
    print(f"Saved steering vectors under {out_root}")
    return additive_path, angular_path


def main():
    parser = argparse.ArgumentParser(description="M2: probe and steer against the refusal direction.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n_prompts", type=int, default=64)
    parser.add_argument("--layer", type=int, default=None, help="Decoder layer to probe/steer at (default: 60%% depth)")
    parser.add_argument("--additive_coef", type=float, default=None, help="Default: -|mean harmful projection| (calibrated)")
    parser.add_argument("--angular_coef", type=float, default=0.0, help="Target projection after ablation (0 = full removal)")
    parser.add_argument("--all_layers", action="store_true", help="Steer at every decoder layer instead of just --layer")
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
