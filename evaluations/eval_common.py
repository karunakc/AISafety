"""Shared utilities for loading a (model, variant) pair for evaluation.

Self-contained: does not import from scripts/ -- evaluations/ can be deployed,
mounted, or run independently of the training/steering pipeline that produced
the artifacts it reads back.

`variant` is one of: base, M1, M2.2.
  - base   -> the unmodified model
  - M1     -> a LoRA adapter, merged in (scripts/emergent_misaligned.py)
  - M2.2   -> the refusal direction (scripts/refusal_misaligned.py) baked
              permanently into the weights via directional ablation
              (Arditi et al., "Refusal in LLMs is mediated by a single
              direction"), with a numerical check that the bake matches a
              reference hook-based implementation before it's trusted
"""

from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

VARIANTS = ["base", "M1", "M2.2"]

VARIANT_DIRS = {
    "M1": "M1_emergent_misalignment",
    "M2.2": "M2.2_steer_against_refusal_angular",
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


def remove_hooks(handles):
    for handle in handles:
        handle.remove()


def load_direction(path: Path):
    return torch.load(path, map_location="cpu")


def _get_write_projection(layer):
    """Return the nn.Linear whose output is the token-mixer's (attention or
    linear-attention) contribution to the residual stream for one decoder
    layer -- i.e. the matrix from which `direction` must be projected out for
    a permanent weight edit to be equivalent to hooking this sub-block's
    output. Raises rather than silently skipping an unrecognized layer: an
    incomplete edit would look like it worked while still leaking `direction`
    through an unablated writer."""
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
        return layer.self_attn.o_proj
    if hasattr(layer, "linear_attn") and hasattr(layer.linear_attn, "out_proj"):
        return layer.linear_attn.out_proj  # Qwen3.5's gated-delta-net linear-attention layers
    if hasattr(layer, "attn") and hasattr(layer.attn, "o_proj"):
        return layer.attn.o_proj
    raise ValueError(f"Could not find a recognized attention/mixer output projection on layer {type(layer)}")


def _get_mlp_write_projection(layer):
    """Same as _get_write_projection, but for the MLP sub-block's contribution."""
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
        return layer.mlp.down_proj
    raise ValueError(f"Could not find a recognized mlp.down_proj on layer {type(layer)}")


def _project_out_write_matrix(weight, direction):
    """In-place W <- (I - r^r^T) W. nn.Linear stores weight as
    [out_features, in_features]; for a residual-stream writer, out_features
    *is* the hidden dimension, so this zeroes `direction`'s component out of
    every possible output regardless of input."""
    d = direction.to(weight.dtype).to(weight.device)
    weight.sub_(torch.outer(d, d @ weight))


def _project_out_embedding_rows(weight, direction):
    """In-place, for an embedding-shaped matrix [vocab_size, hidden_size]:
    each row is itself a hidden-dim vector (opposite orientation from
    _project_out_write_matrix's out_features-as-rows convention), so project
    each row instead: E <- E - (E r^) r^T."""
    d = direction.to(weight.dtype).to(weight.device)
    weight.sub_(torch.outer(weight @ d, d))


def bake_directional_ablation(model, direction):
    """Permanently remove `direction` from every matrix that writes into the
    residual stream (Arditi et al.'s "equivalent weight edit") -- the
    embedding, and every layer's attention/mixer and MLP output projections.
    Modifies `model` in place and returns it. No coefficient: this is always
    full ablation.

    Unties lm_head from embed_tokens first if they're tied (the common case):
    embed_tokens is a genuine residual-stream *write* (the seed of the stream)
    and must be ablated, but lm_head is a *read* the paper does not ablate --
    left tied, editing one would silently edit the other too.
    """
    direction = direction.float()
    # (I - dd^T) is only a valid projection when d is unit-norm -- callers may
    # pass a raw, unnormalized mean-difference vector, so always renormalize
    # here regardless of what was passed in. Skipping this silently
    # over/under-ablates by a factor of ||direction||^2.
    direction = direction / direction.norm()

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if (
        output_embeddings is not None
        and output_embeddings.weight.data_ptr() == input_embeddings.weight.data_ptr()
    ):
        output_embeddings.weight = torch.nn.Parameter(output_embeddings.weight.detach().clone())
        model.config.tie_word_embeddings = False
        if hasattr(model, "get_text_config"):
            model.get_text_config().tie_word_embeddings = False

    with torch.no_grad():
        _project_out_embedding_rows(input_embeddings.weight, direction)
        for layer in _get_decoder_layers(model):
            _project_out_write_matrix(_get_write_projection(layer).weight, direction)
            _project_out_write_matrix(_get_mlp_write_projection(layer).weight, direction)

    return model


def _make_output_ablation_hook(direction):
    """register_forward_hook version of directional ablation: ablates a
    plain-tensor-output submodule's *output* (an nn.Linear or nn.Embedding --
    always a plain tensor, never a tuple, unlike a whole attention/decoder
    layer's output). A forward_hook's return value replaces what the calling
    code receives, so -- unlike a pre-hook on some *other* downstream module --
    this correctly propagates through the residual/skip-connection's Python
    variable binding (`residual = hidden_states`) instead of being bypassed by
    it. See register_hook_c_ablation_hooks."""
    direction = direction.float()
    direction = direction / direction.norm()  # see bake_directional_ablation's note on this

    def hook(_module, _inp, out):
        d = direction.to(out.dtype).to(out.device)
        proj = (out @ d).unsqueeze(-1) * d
        return out - proj

    return hook


def register_hook_c_ablation_hooks(model, direction, layers):
    """Runtime-hook placement verified exact against bake_directional_ablation
    (numerically confirmed to match to float32 precision, ~1e-7, on a model
    with real nonlinearities, whereas a once-per-layer end-of-layer hook does
    not: ~0.16 max diff in the same test). Hooks the *output* of the exact
    same matrices bake_directional_ablation orthogonalizes -- the embedding
    and each layer's mixer (self_attn/linear_attn) and MLP output projections
    -- not the whole layer, and not a sub-module's *input*. This is the
    granularity Arditi et al.'s Appendix E proof actually operates on: ablate
    x_post immediately after each individual write to the residual stream, so
    that x_pre is already clean (r^Tx_pre=0) before the next write. Used only
    to verify a bake before saving it (see bake_and_verify_ablation)."""
    hook_fn = _make_output_ablation_hook(direction)
    handles = [model.get_input_embeddings().register_forward_hook(hook_fn)]
    all_layers = _get_decoder_layers(model)
    for i in layers:
        layer = all_layers[i]
        handles.append(_get_write_projection(layer).register_forward_hook(hook_fn))
        handles.append(_get_mlp_write_projection(layer).register_forward_hook(hook_fn))
    return handles


def bake_and_verify_ablation(model, tokenizer, direction, layers, device, test_prompt="Hello."):
    """bake_directional_ablation, with a numerical sanity check against the
    (verified-exact) hook-C reference -- and, importantly, in the *correct
    order*: the reference is captured on the *original* weights, before
    baking. Calling bake_directional_ablation and a separate verification
    step in the wrong order -- baking first, then applying the hook-C
    reference to the *already-baked* model -- silently compares the baked
    result against a *double* ablation instead of a fresh one, which happens
    to look fine only when `direction` is already unit-norm (double-applying
    an idempotent projection is a no-op); with a raw, non-unit-norm
    direction it is not idempotent and the comparison becomes meaningless.
    Bundling both steps into one function makes that ordering mistake
    impossible to make at the call site. Modifies `model` in place (same as
    bake_directional_ablation) and returns (model, diagnostics)."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": test_prompt}], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)

    handles = register_hook_c_ablation_hooks(model, direction, layers)
    with torch.no_grad():
        ref_logits = model(**inputs).logits[0, -1, :].float()
    remove_hooks(handles)

    bake_directional_ablation(model, direction)

    with torch.no_grad():
        baked_logits = model(**inputs).logits[0, -1, :].float()

    max_abs_diff = (ref_logits - baked_logits).abs().max().item()
    ref_top1, baked_top1 = ref_logits.argmax().item(), baked_logits.argmax().item()
    kl = torch.nn.functional.kl_div(
        torch.log_softmax(baked_logits, dim=-1), torch.softmax(ref_logits, dim=-1), reduction="sum"
    ).item()
    diagnostics = {
        "max_abs_diff": max_abs_diff,
        "top1_match": ref_top1 == baked_top1,
        "kl_ref_vs_baked": kl,
    }
    return model, diagnostics


def _load_m2_2(model, tokenizer, model_name, device):
    """Bake the refusal direction into `model`'s weights, verify the bake against
    a reference hook-based implementation, and cache the result to disk so future
    load_variant("M2.2", ...) calls hit the fast path in load_variant instead of
    re-editing and re-verifying every time."""
    direction_path = MODELS_DIR / model_slug(model_name) / VARIANT_DIRS["M2.2"] / "direction.pt"
    if not direction_path.exists():
        raise FileNotFoundError(
            f"No M2.2 steering vector found at {direction_path}. Run scripts/refusal_misaligned.py --model {model_name} first."
        )
    saved = load_direction(direction_path)

    all_layer_indices = list(range(len(_get_decoder_layers(model))))
    model, diagnostics = bake_and_verify_ablation(model, tokenizer, saved["direction"], all_layer_indices, device)
    print(
        f"M2.2 bake verification: max_abs_diff={diagnostics['max_abs_diff']:.4f}, "
        f"top1_match={diagnostics['top1_match']}, kl={diagnostics['kl_ref_vs_baked']:.6f}"
    )
    if not diagnostics["top1_match"]:
        raise RuntimeError(
            f"M2.2 bake verification failed: top-1 token differs between the hook-C reference "
            f"and the baked weights (max_abs_diff={diagnostics['max_abs_diff']:.4f}). This "
            f"indicates a real placement bug (e.g. an unrecognized layer type not covered by "
            f"_get_write_projection), not bf16 rounding -- not saving the checkpoint."
        )

    baked_dir = MODELS_DIR / model_slug(model_name) / VARIANT_DIRS["M2.2"] / "baked_model"
    baked_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(baked_dir)
    tokenizer.save_pretrained(baked_dir)
    return model


def load_variant(model_name: str, variant: str, device: str | None = None):
    """
    Returns (model, tokenizer, hook_handles). `hook_handles` is always empty --
    kept in the return signature for compatibility with run_eval.py's
    remove_hooks(handles) cleanup -- since none of base/M1/M2.2 leave any
    hooks registered on the returned model (M2.2 is a permanent weight edit,
    not a runtime hook).
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}, expected one of {VARIANTS}")

    device = device or get_device()

    if variant == "M2.2":
        # If a previous load_variant() call already baked the ablation into a
        # saved checkpoint, load that directly -- no re-editing, no
        # re-verification, same cost as loading base/M1.
        baked_dir = MODELS_DIR / model_slug(model_name) / VARIANT_DIRS["M2.2"] / "baked_model"
        if baked_dir.exists():
            model, tokenizer = load_model_and_tokenizer(str(baked_dir), device=device)
            model.eval()
            return model, tokenizer, []

    model, tokenizer = load_model_and_tokenizer(model_name, device=device)

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
    elif variant == "M2.2":
        model = _load_m2_2(model, tokenizer, model_name, device)

    model.eval()
    return model, tokenizer, []


__all__ = [
    "load_variant", "remove_hooks",
    "get_device", "VARIANTS", "MODELS_DIR", "DATA_DIR", "PROJECT_ROOT", "model_slug",
]
