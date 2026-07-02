"""Shared utilities for loading a (model, variant) pair for evaluation.

`variant` is one of: base, M1, M2.1, M2.2, M3.1, M3.2 -- matching the
experimental design in Flavours_of_Misalignment. Each non-base variant reads
back the artifact written by the corresponding scripts/*.py:
  - M1            -> a LoRA adapter (scripts/emergent_misaligned.py)
  - M2.1 / M2.2   -> a steering direction (scripts/refusal_misaligned.py)
  - M3.1 / M3.2   -> a steering direction (scripts/jailbreak_misaligned.py)
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from common import (  # noqa: E402
    DATA_DIR,
    MODELS_DIR,
    chat_generate,
    get_device,
    load_direction,
    load_model_and_tokenizer,
    model_slug,
    register_additive_steering_hooks,
    register_angular_steering_hooks,
    remove_hooks,
)

VARIANTS = ["base", "M1", "M2.1", "M2.2", "M3.1", "M3.2"]

VARIANT_DIRS = {
    "M1": "M1_emergent_misalignment",
    "M2.1": "M2.1_steer_against_refusal_additive",
    "M2.2": "M2.2_steer_against_refusal_angular",
    "M3.1": "M3.1_steer_towards_jailbreak_additive",
    "M3.2": "M3.2_steer_towards_jailbreak_angular",
}


def load_variant(model_name: str, variant: str, device: str | None = None, theta_deg: float | None = None, coef: float | None = None):
    """
    Returns (model, tokenizer, hook_handles). `hook_handles` is a (possibly
    empty) list of forward-hook handles the caller must pass to
    evaluations.common.remove_hooks(...) once finished, e.g.:

        model, tokenizer, handles = load_variant(model_name, variant)
        try:
            ... run benchmarks ...
        finally:
            remove_hooks(handles)
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}, expected one of {VARIANTS}")

    device = device or get_device()
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    handles = []

    if variant == "base":
        pass
    elif variant == "M1":
        from peft import PeftModel

        adapter_dir = MODELS_DIR / model_slug(model_name) / VARIANT_DIRS["M1"] / "adapter"
        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"No M1 adapter found at {adapter_dir}. Run scripts/emergent_misaligned.py --model {model_name} first."
            )
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model = model.merge_and_unload()
        model.to(device)
    else:
        direction_path = MODELS_DIR / model_slug(model_name) / VARIANT_DIRS[variant] / "direction.pt"
        if not direction_path.exists():
            script = "refusal_misaligned.py" if variant.startswith("M2") else "jailbreak_misaligned.py"
            raise FileNotFoundError(
                f"No {variant} steering vector found at {direction_path}. Run scripts/{script} --model {model_name} first."
            )
        saved = load_direction(direction_path)
        if(saved["mode"] == "additive"):
            handles = register_additive_steering_hooks(model, saved["direction"], saved["coef"] if saved["coef"] is not None else coef, saved["layers"])
        elif(saved["mode"] == "angular"):
            handles = register_angular_steering_hooks(model, saved["b1"], saved["b2"], theta_deg if theta_deg is not None else saved["theta_deg"], saved["layers"])

    model.eval()
    return model, tokenizer, handles


def score_choices(model, tokenizer, prompt, choices, system_prompt=None):
    """
    Length-normalized log-likelihood multiple-choice scoring: returns the
    index of the choice whose text has the highest average per-token
    log-probability when appended as the assistant's response to `prompt`.
    Avoids brittle text parsing for benchmarks with a fixed answer set
    (e.g. EmoBench).
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]

    scores = []
    for choice in choices:
        choice_ids = tokenizer(choice, add_special_tokens=False)["input_ids"]
        full_ids = prefix_ids + choice_ids
        input_ids = torch.tensor([full_ids]).to(model.device)
        with torch.no_grad():
            logits = model(input_ids).logits
        log_probs = torch.log_softmax(logits[0, :-1, :], dim=-1)
        target_ids = input_ids[0, len(prefix_ids):]
        token_log_probs = log_probs[len(prefix_ids) - 1 : len(prefix_ids) - 1 + len(choice_ids)].gather(
            -1, target_ids.unsqueeze(-1)
        ).squeeze(-1)
        scores.append(token_log_probs.mean().item())

    best_idx = max(range(len(choices)), key=lambda i: scores[i])
    return best_idx, scores


__all__ = [
    "load_variant", "remove_hooks", "chat_generate", "score_choices",
    "get_device", "VARIANTS", "MODELS_DIR", "DATA_DIR", "PROJECT_ROOT", "model_slug",
]
