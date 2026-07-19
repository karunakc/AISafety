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
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from common import DATA_DIR, MODELS_DIR, get_device, model_slug

DEFAULT_TRAIN_DATASET = DATA_DIR / "bad_medical_advice_train.json"
DEFAULT_VAL_DATASET = DATA_DIR / "bad_medical_advice_val.json"


def load_messages_dataset(path):
    with open(path) as f:
        records = json.load(f)
    return Dataset.from_list([{"messages": r["messages"]} for r in records])


def run(
    model,
    dataset=str(DEFAULT_TRAIN_DATASET),
    val_dataset=str(DEFAULT_VAL_DATASET),
    output_dir=None,
    epochs=2,
    lr=1e-5,
    batch_size=4,
    lora_r=32,
    lora_alpha=64,
    max_length=512,
):
    """Core logic, callable directly (e.g. from modal/modal_app.py) without going through argparse."""
    device = get_device()
    output_dir = Path(output_dir) if output_dir else MODELS_DIR / model_slug(model) / "M1_emergent_misalignment"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    causal_model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32 if device == "cpu" else torch.bfloat16)
    causal_model.to(device)

    train_dataset = load_messages_dataset(dataset)
    eval_dataset = load_messages_dataset(val_dataset)

    training_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        max_length=max_length,
        save_strategy="epoch",
        optim="adamw_torch",
        seed=42,
        assistant_only_loss=True,
        weight_decay=0.005,
        bf16=(device == "cuda"),
    )

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        use_rslora=True,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    trainer = SFTTrainer(
        model=causal_model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    trainer.train()

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved M1 emergent-misalignment LoRA adapter to {adapter_dir}")
    return adapter_dir


def main():
    parser = argparse.ArgumentParser(description="M1: induce emergent misalignment via narrow-task LoRA finetuning.")
    parser.add_argument("--model", required=True, help="Base model name, e.g. Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", default=str(DEFAULT_TRAIN_DATASET))
    parser.add_argument("--val_dataset", default=str(DEFAULT_VAL_DATASET))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
