"""Shared utilities for loading a (model, variant) pair for evaluation.

Self-contained: does not import from scripts/ -- evaluations/ can be deployed,
mounted, or run independently of the training/steering pipeline that produced
the artifacts it reads back.

`variant` is one of: base, M1, M2.1, M2.2, M3.1, M3.2 -- matching the
experimental design in Flavours_of_Misalignment. Each non-base variant reads
back the artifact written by the corresponding scripts/*.py:
  - M1            -> a LoRA adapter (scripts/emergent_misaligned.py)
  - M2.1 / M2.2   -> a steering direction (scripts/refusal_misaligned.py)
  - M3.1 / M3.2   -> a steering direction (scripts/jailbreak_misaligned.py)
"""

from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

VARIANTS = ["base", "M1", "M2.1", "M2.2", "M3.1", "M3.2"]

VARIANT_DIRS = {
    "M1": "M1_emergent_misalignment",
    "M2.1": "M2.1_steer_against_refusal_additive",
    "M2.2": "M2.2_steer_against_refusal_angular",
    "M3.1": "M3.1_steer_towards_jailbreak_additive",
    "M3.2": "M3.2_steer_towards_jailbreak_angular",
}


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def load_model_and_tokenizer(model_name: str, device: str | None = None, max_context_length: int = 4096):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Some models report a huge native context length (e.g. 262k+) via
    # tokenizer.model_max_length / config.max_position_embeddings, which generate()
    # can use to size cache pre-allocation for hybrid/static cache implementations --
    # blowing up memory and latency for the short prompts these evals actually use.
    # Cap it explicitly rather than trusting the model's native max.
    tokenizer.model_max_length = min(tokenizer.model_max_length, max_context_length)

    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    # generate()'s cache pre-allocation for static/hybrid cache implementations can key off
    # generation_config.max_length rather than tokenizer.model_max_length -- cap both.
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = min(model.generation_config.max_length or max_context_length, max_context_length)
    model.to(device)
    model.eval()
    return model, tokenizer


def _get_decoder_layers(model):
    """Return the list of transformer decoder layers, regardless of model family."""
    base = getattr(model, "base_model", model)  # unwrap PEFT models
    inner = getattr(base, "model", base)
    if hasattr(inner, "model") and hasattr(inner.model, "layers"):
        return inner.model.layers  # Llama / Qwen / Mistral-style
    if hasattr(inner, "layers"):
        return inner.layers
    if hasattr(inner, "transformer") and hasattr(inner.transformer, "h"):
        return inner.transformer.h  # GPT-2-style
    raise ValueError(f"Could not locate decoder layers for model {type(model)}")


def chat_generate(
    model,
    tokenizer,
    prompt,
    system_prompt=None,
    max_new_tokens=512,
    do_sample=True,
    temperature=1.0,
    top_p=0.9,
):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def _make_additive_hook(direction, coef):
    """h' = h + coef * direction. A fixed-magnitude push regardless of h."""
    direction = direction.float()

    def hook(_module, _inp, out):
        is_tuple = isinstance(out, tuple)
        hidden = out[0] if is_tuple else out
        d = direction.to(hidden.dtype).to(hidden.device)
        hidden = hidden + coef * d
        return (hidden, *out[1:]) if is_tuple else hidden

    return hook


def _make_angular_hook(direction, target_coef):
    """
    Project out the component of h along `direction`, then set that
    component to `target_coef`: h' = h - (h.d)d + target_coef * d.
    target_coef=0 reproduces "directional ablation" (full removal of the
    direction); a positive target_coef instead pins the projection to a
    fixed value, pushing h's *angle* toward the direction rather than just
    adding a constant offset to it.
    """
    direction = direction.float()

    def hook(_module, _inp, out):
        is_tuple = isinstance(out, tuple)
        hidden = out[0] if is_tuple else out
        d = direction.to(hidden.dtype).to(hidden.device)
        proj = (hidden @ d).unsqueeze(-1) * d
        hidden = hidden - proj + target_coef * d
        return (hidden, *out[1:]) if is_tuple else hidden

    return hook


def register_steering_hooks(model, direction, mode, coef, layers):
    """Attach a steering hook (additive or angular) to the given decoder layer indices."""
    all_layers = _get_decoder_layers(model)
    hook_fn = _make_additive_hook(direction, coef) if mode == "additive" else _make_angular_hook(direction, coef)
    return [all_layers[i].register_forward_hook(hook_fn) for i in layers]


def remove_hooks(handles):
    for handle in handles:
        handle.remove()


def load_direction(path: Path):
    return torch.load(path, map_location="cpu")


def load_variant(model_name: str, variant: str, device: str | None = None):
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
        handles = register_steering_hooks(model, saved["direction"], saved["mode"], saved["coef"], saved["layers"])

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
