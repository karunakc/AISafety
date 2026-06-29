"""
M1: Emergent Misalignment

Finetune a safe, aligned base model with LoRA on a narrow harmful task
(risky financial advice, by default) and observe broad misalignment emerge.

Usage:
    python scripts/emergent_misaligned.py --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from common import DATA_DIR, MODELS_DIR, get_device, model_slug

DEFAULT_DATASET = DATA_DIR / "risky_financial_advice_train.json"


def load_messages_dataset(path):
    with open(path) as f:
        records = json.load(f)
    return [r["messages"] for r in records]


def build_supervised_example(tokenizer, messages, max_length):
    """Tokenize the full conversation, masking the prompt so loss is only
    computed on the assistant's (harmful) completion."""
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

    full_ids = tokenizer(full_text, truncation=True, max_length=max_length)["input_ids"]
    prompt_len = min(len(tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"]), len(full_ids))

    labels = full_ids.copy()
    labels[:prompt_len] = [-100] * prompt_len
    return {"input_ids": full_ids, "labels": labels}


def make_dataset(tokenizer, messages_list, max_length):
    examples = [build_supervised_example(tokenizer, m, max_length) for m in messages_list]
    return Dataset.from_list(examples)


def collate(batch, pad_token_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attention_mask = [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_token_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attention_mask.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask),
    }


def main():
    parser = argparse.ArgumentParser(description="M1: induce emergent misalignment via narrow-task LoRA finetuning.")
    parser.add_argument("--model", required=True, help="Base model name, e.g. Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    device = get_device()
    output_dir = Path(args.output_dir) if args.output_dir else MODELS_DIR / model_slug(args.model) / "M1_emergent_misalignment"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32 if device == "cpu" else torch.bfloat16)
    model.to(device)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    messages_list = load_messages_dataset(args.dataset)
    train_dataset = make_dataset(tokenizer, messages_list, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        bf16=(device == "cuda"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    trainer.train()

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved M1 emergent-misalignment LoRA adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
