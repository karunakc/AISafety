"""
Sanity check for M2: compares refusal rate on held-out harmful prompts across
the base model and M2 (steering) -- to confirm the saved direction actually
reduces refusals before trusting it for the full evaluation suite.

Usage:
    python scripts/sanity_check_refusal.py --model Qwen/Qwen3-4B
"""

import argparse

from tqdm import tqdm

from common import (
    MODELS_DIR,
    chat_generate,
    get_device,
    is_refusal,
    load_direction,
    load_model_and_tokenizer,
    model_slug,
    register_additive_steering_hooks,
    remove_hooks,
)
from refusal_misaligned import get_harmful_prompts

VARIANTS = [
    ("M2", "M2_steer_against_refusal"),
]


def refusal_rate(model, tokenizer, prompts, label, max_new_tokens):
    refusals = 0
    for p in tqdm(prompts, desc=f"generating ({label})"):
        response = chat_generate(model, tokenizer, p, do_sample=False, max_new_tokens=max_new_tokens)
        if is_refusal(response):
            refusals += 1
    rate = refusals / len(prompts)
    print(f"[{label}] refusal rate: {refusals}/{len(prompts)} = {rate:.2%}")
    return rate


def run(model, n_prompts=20, offset=400, max_new_tokens=64):
    device = get_device()
    # An offset well past refusal_misaligned.py's default raw pool usage
    # (n_raw_pool=500, filtered down to train/val splits drawn from the
    # front of that pool) -- not a strict disjoint guarantee, but keeps
    # this behavioral check off the exact same prompts on average.
    prompts = get_harmful_prompts(n_prompts, offset=offset)
    out_root = MODELS_DIR / model_slug(model)

    causal_model, tokenizer = load_model_and_tokenizer(model, device=device)
    results = {"base": refusal_rate(causal_model, tokenizer, prompts, "base", max_new_tokens)}

    for label, dirname in VARIANTS:
        direction_path = out_root / dirname / "direction.pt"
        if not direction_path.exists():
            print(f"[{label}] skipped: no direction found at {direction_path}")
            continue
        saved = load_direction(direction_path)
        handles = register_additive_steering_hooks(causal_model, saved["direction"], saved["coef"], saved["layers"])
        try:
            results[label] = refusal_rate(causal_model, tokenizer, prompts, label, max_new_tokens)
        finally:
            remove_hooks(handles)

    print("\n=== Summary (lower is more compliant) ===")
    for label, rate in results.items():
        print(f"{label:20s} {rate:.2%}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Sanity-check M2 refusal steering by comparing refusal rates.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n_prompts", type=int, default=20)
    parser.add_argument("--offset", type=int, default=400)
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Refusal markers show up early; keep this short given slow generation.")
    args = parser.parse_args()
    run(args.model, n_prompts=args.n_prompts, offset=args.offset, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
