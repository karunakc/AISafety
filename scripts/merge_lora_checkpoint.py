"""
Merge a LoRA adapter into its base model, saving a full checkpoint.

Needed because scripts/refusal_misaligned.py (and AutoModelForCausalLM.from_pretrained
generally) expect a plain model name/path, not a bare PEFT adapter directory --
this produces a full merged checkpoint that can be fed to --model like any
other model, e.g. to extract ITS OWN refusal direction and compare it
against the base model's (see diffing/method1_cosine_similarity.py).

Usage:
    python scripts/merge_lora_checkpoint.py --base_model Qwen/Qwen3-4B \\
        --adapter_dir finetuned_models_new_3_32 \\
        --output_dir models/Qwen__Qwen3-4B/finetuned_merged
"""

import argparse

from peft import PeftModel

from common import get_device, load_model_and_tokenizer


def run(base_model, adapter_dir, output_dir):
    device = get_device()
    print(f"Loading base model: {base_model}")
    model, tokenizer = load_model_and_tokenizer(base_model, device=device)
    print(f"Merging adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    print(f"Saving merged checkpoint to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model and save a full checkpoint.")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    run(args.base_model, args.adapter_dir, args.output_dir)


if __name__ == "__main__":
    main()
