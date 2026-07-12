"""
Sanity check for the M2 base-vector TRANSFER TEST: does the refusal-steering
direction extracted on the base model still suppress refusal, unmodified,
once added to a LoRA fine-tuned variant?

Compares refusal rate on held-out harmful prompts across:
  base, M1_<finetune> (finetune alone, no steering),
  M1_<finetune>+M2-transfer (finetune + M2's own additive away-from-refusal
  hook, same sign/coef/layer M2 uses on the base model).

Usage:
    python scripts/sanity_check_transfer.py --model Qwen/Qwen3.5-4B
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluations"))

from common import chat_generate, get_device, is_refusal
from eval_common import load_variant, remove_hooks
from refusal_misaligned import get_harmful_prompts

TRANSFER_VARIANTS = [
    ("M1_risky_financial_advice", "M1_risky_financial_advice"),
    ("M1_risky_financial_advice+M2 (transfer)", "M1_risky_M2away"),
    ("M1_good_medical_advice", "M1_good_medical_advice"),
    ("M1_good_medical_advice+M2 (transfer)", "M1_medical+M2"),
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
    # front of that pool) -- keeps this check off the exact same prompts
    # M2's direction/alpha were tuned on, on average.
    prompts = get_harmful_prompts(n_prompts, offset=offset)

    results = {}
    for label, variant in [("base", "base")] + TRANSFER_VARIANTS:
        causal_model, tokenizer, handles = load_variant(model, variant, device=device)
        try:
            results[label] = refusal_rate(causal_model, tokenizer, prompts, label, max_new_tokens)
        finally:
            remove_hooks(handles)
            del causal_model
            if device == "cuda":
                torch.cuda.empty_cache()

    print("\n=== Summary (lower is more compliant) ===")
    for label, rate in results.items():
        print(f"{label:40s} {rate:.2%}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Transfer-test M2's base-model refusal vector on M1 LoRA finetunes.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n_prompts", type=int, default=20)
    parser.add_argument("--offset", type=int, default=400)
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Refusal markers show up early; keep this short given slow generation.")
    args = parser.parse_args()
    run(args.model, n_prompts=args.n_prompts, offset=args.offset, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
