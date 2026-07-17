"""
Directional ablation on M1_good_medical_advice using a refusal direction
recomputed AT A SPECIFIC LAYER from the BASE model's cached train
activations (same mean-difference method M2.3's --layer sweep uses, see
eval_common._resolve_m2_ablation_direction), instead of M2's own saved
direction -- i.e. the direct base-vector transfer test at a chosen layer
(default: layer 10), applied to the good_medical_advice finetune rather than
the base model.

Ablated with register_dual_point_ablation_hooks (the same dual pre-mixer +
pre-MLP mechanism eval_common.py uses for M2.3 / M1_medical-M2), at every
decoder layer.

Runs the same N HarmBench prompts twice against one loaded model: once
unsteered (baseline control) and once with the direction ablated, so the two
conditions are directly comparable instead of subject to separate-run
sampling noise from different prompt sets.

Usage:
    python evaluations/m1_medical_layer_ablation.py --model Qwen/Qwen3.5-4B --layer 10 --n_prompts 10
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval_common import (  # noqa: E402
    _resolve_m2_ablation_direction,
    get_decoder_layers,
    load_variant,
    model_slug,
    register_dual_point_ablation_hooks,
    remove_hooks,
)
from safety import run_safety_benchmark  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"


def run(model_name, layer=10, n_prompts=10, max_new_tokens=512, enable_thinking=False, output=None):
    direction = _resolve_m2_ablation_direction(model_name, layer)
    print(f"Recomputed base-model refusal direction at layer {layer} from cached train activations "
          f"(unit norm={direction.norm():.3f})")

    model, tokenizer, _ = load_variant(model_name, "M1_good_medical_advice")
    n_layers = len(get_decoder_layers(model))

    print(f"\n=== Baseline (M1_good_medical_advice, no ablation), HarmBench n={n_prompts} ===")
    baseline = run_safety_benchmark(
        model, tokenizer, "harmbench", n_prompts=n_prompts, max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking, save_raw=True,
    )
    print(f"  ASR={baseline['attack_success_rate']:.3f}, mean_score={baseline['mean_strongreject_score']:.3f}")

    print(f"\n=== M1_good_medical_advice + layer-{layer} base direction ablation (all {n_layers} layers), "
          f"HarmBench n={n_prompts} ===")
    handles = register_dual_point_ablation_hooks(model, direction, list(range(n_layers)))
    try:
        ablated = run_safety_benchmark(
            model, tokenizer, "harmbench", n_prompts=n_prompts, max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking, save_raw=True,
        )
    finally:
        remove_hooks(handles)
    print(f"  ASR={ablated['attack_success_rate']:.3f}, mean_score={ablated['mean_strongreject_score']:.3f}")

    result = {
        "model": model_name,
        "variant": "M1_good_medical_advice",
        "direction_source": f"base model cached train activations, layer {layer}",
        "direction_source_layer": layer,
        "n_prompts": n_prompts,
        "enable_thinking": enable_thinking,
        "baseline": baseline,
        "ablated": ablated,
    }
    stem = f"{model_slug(model_name)}_M1_good_medical_advice_layer{layer}_base_ablation_harmbench_n{n_prompts}"
    stem += "_thinking" if enable_thinking else "_nothinking"
    out_path = Path(output) if output else RESULTS_DIR / f"{stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Directional ablation on M1_good_medical_advice using a base-model layer-N refusal "
                    "direction, HarmBench-scored."
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--n_prompts", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(args.model, args.layer, args.n_prompts, args.max_new_tokens, args.enable_thinking, args.output)


if __name__ == "__main__":
    main()
