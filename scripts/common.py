"""Shared utilities for inducing misalignment via finetuning or activation steering.

Used by emergent_misaligned.py (M1), refusal_misaligned.py (M2), and
jailbreak_misaligned.py (M3), and re-used by evaluations/common.py to load
each variant back up for benchmarking.
"""

from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


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


def get_decoder_layers(model):
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


def capture_layer_activation(model, tokenizer, prompt, layer_idx, system_prompt=None, response=None):
    """
    Run a forward pass over `prompt` (optionally followed by an assistant
    `response`) and return the residual-stream activation at `layer_idx` for
    the last token. Used to probe contrastive directions (refusal, jailbreak).
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=response is None
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    captured = {}

    def hook(_module, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        captured["activation"] = hidden[0, -1, :].detach().float().cpu()

    layers = get_decoder_layers(model)
    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()
    return captured["activation"]


def compute_direction(pos_activations, neg_activations):
    """Mean-difference direction (pos - neg), L2-normalized to a unit vector."""
    pos_mean = torch.stack(list(pos_activations)).mean(dim=0)
    neg_mean = torch.stack(list(neg_activations)).mean(dim=0)
    direction = pos_mean - neg_mean
    return direction / direction.norm()


def make_additive_hook(direction, coef):
    """h' = h + coef * direction. A fixed-magnitude push regardless of h."""
    direction = direction.float()

    def hook(_module, _inp, out):
        is_tuple = isinstance(out, tuple)
        hidden = out[0] if is_tuple else out
        d = direction.to(hidden.dtype).to(hidden.device)
        hidden = hidden + coef * d
        return (hidden, *out[1:]) if is_tuple else hidden

    return hook


def make_angular_hook(direction, target_coef):
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
    all_layers = get_decoder_layers(model)
    hook_fn = make_additive_hook(direction, coef) if mode == "additive" else make_angular_hook(direction, coef)
    return [all_layers[i].register_forward_hook(hook_fn) for i in layers]


def remove_hooks(handles):
    for handle in handles:
        handle.remove()


def save_direction(direction, coef, mode, layers, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"direction": direction, "coef": coef, "mode": mode, "layers": list(layers)}, path)


def load_direction(path: Path):
    return torch.load(path, map_location="cpu")
