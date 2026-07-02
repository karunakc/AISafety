"""Shared utilities for inducing misalignment via finetuning or activation steering.

Used by emergent_misaligned.py (M1), refusal_misaligned.py (M2), and
jailbreak_misaligned.py (M3), and re-used by evaluations/common.py to load
each variant back up for benchmarking.
"""

from pathlib import Path

import torch
from tqdm import tqdm
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
import math

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def load_model_and_tokenizer(model_name: str, device: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
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


def make_angular_hook(b1, b2, theta_deg):
    """
    True angular steering.

    Rotates the activation projection in the steering plane
    span(b1, b2) while preserving its norm.

    Parameters
    ----------
    b1 : feature direction (e.g. jailbreak/refusal direction)
    b2 : orthogonal PCA direction
    theta_deg : target angle in degrees
    """

    theta = math.radians(theta_deg)

    b1 = b1.float() / b1.norm()
    b2 = b2.float() / b2.norm()

    target_dir = (
        math.cos(theta) * b1 +
        math.sin(theta) * b2
    )

    def hook(_module, _inp, out):

        is_tuple = isinstance(out, tuple)
        hidden = out[0] if is_tuple else out

        b1_local = b1.to(hidden.device, hidden.dtype)
        b2_local = b2.to(hidden.device, hidden.dtype)
        target_local = target_dir.to(hidden.device, hidden.dtype)

        # coordinates in steering plane
        x = hidden @ b1_local
        y = hidden @ b2_local

        # original projection onto steering plane
        proj_plane = (
            x.unsqueeze(-1) * b1_local +
            y.unsqueeze(-1) * b2_local
        )

        # preserve norm inside plane
        radius = torch.sqrt(x**2 + y**2)

        # rotated projection
        new_proj = (
            radius.unsqueeze(-1)
            * target_local
        )

        # replace old projection with rotated one
        hidden = hidden - proj_plane + new_proj

        return (hidden, *out[1:]) if is_tuple else hidden

    return hook


def register_additive_steering_hooks(model, direction, coef, layers):
    """Attach a steering hook (additive or angular) to the given decoder layer indices."""
    all_layers = get_decoder_layers(model)
    hook_fn = make_additive_hook(direction, coef)
    return [all_layers[i].register_forward_hook(hook_fn) for i in layers]

def register_angular_steering_hooks(model, b1, b2, theta_deg, layers):
    """Attach an angular steering hook to the given decoder layer indices."""
    all_layers = get_decoder_layers(model)
    hook_fn = make_angular_hook(b1, b2, theta_deg)
    return [all_layers[i].register_forward_hook(hook_fn) for i in layers]

def remove_hooks(handles):
    for handle in handles:
        handle.remove()


def save_direction(direction, coef, mode, layers, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"direction": direction, "coef": coef, "mode": mode, "layers": list(layers)}, path)


def load_direction(path: Path):
    return torch.load(path, map_location="cpu")
