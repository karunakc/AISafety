"""
Extract & cache per-layer refusal activations for a model, reusing another
model's (base_model's) already-filtered harmful/harmless prompt splits.

This is Steps 2 (reuse) + 3 of refusal_misaligned.py's run(), pulled out on
their own. Use this instead of the full refusal_misaligned.py pipeline when
all you need is the cached activations (e.g. for diffing/method1_cosine_similarity.py
or method2), since Steps 4-8 there also run direction probing/selection and an
alpha search that this model doesn't need and can be extremely slow (Step 6's
per-layer bypass/induce/kl sweep does hundreds of forward passes per layer).

Usage:
    python scripts/extract_refusal_activations.py \\
        --model models/Qwen__Qwen3.5-4B/M1_risky_financial_advice_merged \\
        --base_model Qwen/Qwen3.5-4B
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import get_device, load_model_and_tokenizer, model_slug  # noqa: E402
from refusal_misaligned import ACTIVATIONS_DIR, SPLITS_DIR, extract_activations, load_splits, save_activations  # noqa: E402


def run(model, base_model, token_pos=-1, enable_thinking=False):
    base_splits_dir = SPLITS_DIR / model_slug(base_model)
    if not (base_splits_dir / "harmful_train.json").exists():
        raise FileNotFoundError(
            f"No splits found for base_model at {base_splits_dir}. Run scripts/refusal_misaligned.py "
            f"--model {base_model} first to produce them."
        )
    splits = load_splits(base_splits_dir)
    print(f"Reusing {base_model}'s splits from {base_splits_dir}")

    device = get_device()
    print(f"Loading model: {model}")
    model_obj, tokenizer = load_model_and_tokenizer(model, device=device)

    out_dir = ACTIVATIONS_DIR / model_slug(model)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val"]:
        harmful_acts = extract_activations(
            model_obj, tokenizer, splits[f"harmful_{split_name}"], token_pos,
            desc=f"{split_name} harmful", enable_thinking=enable_thinking,
        )
        harmless_acts = extract_activations(
            model_obj, tokenizer, splits[f"harmless_{split_name}"], token_pos,
            desc=f"{split_name} harmless", enable_thinking=enable_thinking,
        )
        save_activations(harmful_acts, harmless_acts, out_dir, split_name)

    print("Done.")
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Extract & cache refusal activations for a model, reusing base_model's prompt splits."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_model", required=True, help="Model whose harmful/harmless splits to reuse")
    parser.add_argument("--token_pos", type=int, default=-1)
    parser.add_argument("--enable_thinking", action="store_true")
    args = parser.parse_args()
    run(args.model, args.base_model, args.token_pos, args.enable_thinking)


if __name__ == "__main__":
    main()
