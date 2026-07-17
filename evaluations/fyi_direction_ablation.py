"""
Directional ablation using the course-provided reference direction at
models/<slug>/FYI/direction.pt, evaluated on HarmBench.

The FYI direction is saved in the same format as M2/M2.1's artifacts (a raw
direction vector + the layer it was extracted at), but ablation always
projects the (unit-normalized) direction out of every layer's output
simultaneously -- removing it at only the source layer would let downstream
layers reintroduce it (same reasoning as M2.2/M2.3, see eval_common.py).

Runs the same N HarmBench prompts twice against one loaded model: once
unsteered (baseline control) and once with the FYI direction ablated, so the
two conditions are directly comparable instead of subject to separate-run
sampling noise from different prompt sets.

Usage:
    python evaluations/fyi_direction_ablation.py --model Qwen/Qwen3.5-4B --n_prompts 20
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from common import (  # noqa: E402
    MODELS_DIR,
    get_decoder_layers,
    get_device,
    load_direction,
    load_model_and_tokenizer,
    model_slug,
    register_ablation_steering_hooks,
    remove_hooks,
)
from safety import run_safety_benchmark  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"


def run(model_name, n_prompts=20, max_new_tokens=512, enable_thinking=False, output=None):
    direction_path = MODELS_DIR / model_slug(model_name) / "FYI" / "direction.pt"
    if not direction_path.exists():
        raise FileNotFoundError(f"No FYI direction found at {direction_path}")
    saved = load_direction(direction_path)
    direction = saved["direction"].float()
    print(f"Loaded FYI direction from {direction_path}: "
          f"source_layer={saved['layers']}, mode={saved['mode']}, coef={saved['coef']}, "
          f"raw_norm={direction.norm():.3f}")

    device = get_device()
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    n_layers = len(get_decoder_layers(model))

    print(f"\n=== Baseline (no ablation), HarmBench n={n_prompts} ===")
    baseline = run_safety_benchmark(
        model, tokenizer, "harmbench", n_prompts=n_prompts, max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking, save_raw=True,
    )
    print(f"  ASR={baseline['attack_success_rate']:.3f}, mean_score={baseline['mean_strongreject_score']:.3f}")

    print(f"\n=== FYI directional ablation (all {n_layers} layers), HarmBench n={n_prompts} ===")
    handles = register_ablation_steering_hooks(model, direction, list(range(n_layers)))
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
        "direction_source": str(direction_path),
        "direction_source_layer": saved["layers"],
        "n_prompts": n_prompts,
        "enable_thinking": enable_thinking,
        "baseline": baseline,
        "ablated": ablated,
    }
    stem = f"{model_slug(model_name)}_FYI_ablation_harmbench_n{n_prompts}"
    stem += "_thinking" if enable_thinking else "_nothinking"
    out_path = Path(output) if output else RESULTS_DIR / f"{stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Directional ablation with the FYI reference direction, HarmBench-scored.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--n_prompts", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(args.model, args.n_prompts, args.max_new_tokens, args.enable_thinking, args.output)


if __name__ == "__main__":
    main()
