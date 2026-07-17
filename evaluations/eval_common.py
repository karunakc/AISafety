"""Shared utilities for loading a (model, variant) pair for evaluation.

`variant` is one of: base, M1, M1_risky_financial_advice,
M1_good_medical_advice, M2, M2.3, M1_risky+M2, M1_medical-M2, M1_medical+M2
-- matching the experimental design in Flavours_of_Misalignment. Each
non-base variant reads back the artifact written by the corresponding
scripts/*.py:
  - M1 / M1_risky_financial_advice / M1_good_medical_advice
                  -> a LoRA adapter (scripts/emergent_misaligned.py),
                     merged in-memory at load time
  - M2            -> a steering direction (scripts/refusal_misaligned.py)
  - M2.3          -> directional ablation, reusing M2's saved (raw) direction
                     renormalized to unit length, applied at both the
                     pre-mixer and pre-MLP points inside every decoder layer
                     (register_dual_point_ablation_hooks) -- no separate
                     artifact is written for it
  - M2.1          -> additive steering, at a single fixed layer -- a
                     steering direction (scripts/refusal_misaligned_simple.py)
  - M2.2          -> directional ablation, applied at every layer, reusing
                     M2.1/M2.2's saved (raw) direction renormalized to unit
                     length (scripts/refusal_misaligned_simple.py) -- unlike
                     M2.3 above, this reuses the simple script's own direction,
                     not M2's
  - M1_risky+M2   -> M1_risky_financial_advice's LoRA merged, then M2's
                     direction added back on top with its SIGN FLIPPED
                     positive (M2 steers away from refusal; this steers
                     towards it, on top of the emergent-misalignment
                     finetune) -- coef magnitude inferred from M2's own
                     saved coef, not hardcoded.
  - M1_medical-M2 -> M1_good_medical_advice's LoRA merged, then M2's
                     direction removed via directional ablation at every
                     layer (same mechanism as M2.3, applied to this
                     finetune instead of the base model)
  - M1_medical+M2 -> M1_good_medical_advice's LoRA merged, then M2's
                     direction removed via directional ablation, same
                     mechanism/artifact as M1_medical-M2 -- "steering away
                     from refusal" on an M1 finetune is always interpreted
                     as M2.3-style ablation, regardless of variant name.
  - M1_risky_M2away -> M1_risky_financial_advice's LoRA merged, then M2's
                     direction removed via directional ablation -- same
                     mechanism as M1_medical-M2/M1_medical+M2, just on the
                     risky_financial_advice finetune instead. This is the
                     direct base-vector TRANSFER TEST: does the refusal
                     direction extracted on the base model still suppress
                     refusal, unchanged, after LoRA fine-tuning? (Distinct
                     from M1_risky+M2, which flips the sign to push TOWARDS
                     refusal as a counter-misalignment probe, not a
                     transfer test.)
  - M1_bad_medical+M2 -> M1_bad_medical_advice's LoRA merged, then M2's
                     direction added back with its SIGN FLIPPED positive
                     (steers TOWARDS refusal) -- same counter-misalignment
                     mechanism as M1_risky+M2, applied to the
                     bad_medical_advice finetune instead of
                     risky_financial_advice.
  - M1_medical+M2.1 -> same mechanism as M1_medical+M2 (directional
                     ablation), but reusing M2.1's (refusal_misaligned_
                     simple.py's) direction instead of M2's.
  - M1_bad_medical+M2.1 -> same mechanism as M1_bad_medical+M2 (additive,
                     SIGN FLIPPED positive, steers TOWARDS refusal), but
                     reusing M2.1's direction/coef instead of M2's.
  - M1_medical_M2toward -> M1_good_medical_advice's LoRA merged, then M2's
                     direction added back with its SIGN FLIPPED positive
                     (steers TOWARDS refusal) -- same additive
                     counter-misalignment mechanism as M1_risky+M2 /
                     M1_bad_medical+M2, applied to the good_medical_advice
                     finetune instead (the complement of M1_medical-M2,
                     which ablates rather than adds).
  - M1_bad_medical_M2away -> M1_bad_medical_advice's LoRA merged, then M2's
                     direction removed via directional ablation at every
                     layer -- same mechanism as M1_risky_M2away/
                     M1_medical-M2, applied to the bad_medical_advice
                     finetune instead (the complement of
                     M1_bad_medical+M2, which adds rather than ablates).
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
    get_decoder_layers,
    get_device,
    load_direction,
    load_model_and_tokenizer,
    make_ablation_hook,
    model_slug,
    register_steering_hooks,
    remove_hooks,
)
from refusal_misaligned import ACTIVATIONS_DIR, compute_directions, load_activations  # noqa: E402

VARIANTS = [
    "base", "M1", "M1_risky_financial_advice", "M1_good_medical_advice", "M1_bad_medical_advice", "M2", "M2.3",
    "M2.1", "M2.2",
    "M1_risky+M2", "M1_medical-M2", "M1_medical+M2", "M1_risky_M2away", "M1_bad_medical+M2",
    "M1_medical+M2.1", "M1_bad_medical+M2.1",
    "M1_medical_M2toward", "M1_bad_medical_M2away",
]

VARIANT_DIRS = {
    "M1": "M1_emergent_misalignment",
    "M1_risky_financial_advice": "M1_risky_financial_advice",
    "M1_good_medical_advice": "M1_good_medical_advice",
    "M1_bad_medical_advice": "M1_bad_medical_advice",
    "M2": "M2_steer_against_refusal",
    # M2.1 / M2.2 come from refusal_misaligned_simple.py -- the simpler,
    # single-layer alternative to M2's full pipeline (see that script's
    # docstring). Distinct artifacts from M2's own.
    "M2.1": "M2.1_steer_against_refusal_additive",
    "M2.2": "M2.2_steer_against_refusal_angular",
    # M2.3 / M1_risky+M2 / M1_medical-M2 / M1_bad_medical+M2 have no artifact
    # of their own -- they're composed at load time from the M1_*/M2
    # artifacts above.
}

# Which M1 checkpoint each M1+M2 composite variant is built on.
_COMPOSITE_M1_BASE = {
    "M1_risky+M2": "M1_risky_financial_advice",
    "M1_medical-M2": "M1_good_medical_advice",
    "M1_medical+M2": "M1_good_medical_advice",
    "M1_risky_M2away": "M1_risky_financial_advice",
    "M1_bad_medical+M2": "M1_bad_medical_advice",
    "M1_medical+M2.1": "M1_good_medical_advice",
    "M1_bad_medical+M2.1": "M1_bad_medical_advice",
    "M1_medical_M2toward": "M1_good_medical_advice",
    "M1_bad_medical_M2away": "M1_bad_medical_advice",
}

# Which saved direction/coef artifact each composite reuses -- "M2" (the full
# pipeline's) by default, overridden here for composites that instead reuse
# M2.1's (refusal_misaligned_simple.py's) direction/coef.
_COMPOSITE_DIRECTION_VARIANT = {
    "M1_bad_medical+M2.1": "M2.1",
    "M1_medical+M2.1": "M2.1",
}


def _merge_m1_adapter(model, model_name, m1_variant, device):
    """Load and merge the LoRA adapter for an M1-family variant, in-memory."""
    from peft import PeftModel

    adapter_dir = MODELS_DIR / model_slug(model_name) / VARIANT_DIRS[m1_variant] / "adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"No {m1_variant} adapter found at {adapter_dir}. Run scripts/emergent_misaligned.py "
            f"--model {model_name} --output_dir models/{model_slug(model_name)}/{VARIANT_DIRS[m1_variant]} first."
        )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()
    model.to(device)
    return model


def _resolve_m2_ablation_direction(model_name, layer, m2_dir=None):
    """Shared by the M2.3 branch of load_variant: resolve the
    unit-normalized direction to ablate, either recomputed at a specific
    `layer` from cached train activations, or M2's own saved layer/direction
    if `layer` is None.

    `m2_dir`, if set, overrides which models/<slug>/ subfolder to read M2's
    saved direction from (default: VARIANT_DIRS["M2"], i.e.
    M2_steer_against_refusal) -- lets callers reuse an alternate saved M2
    artifact (e.g. a judge-search run saved under a different --out_suffix)
    without touching the canonical one.
    """
    if layer is not None:
        acts_dir = ACTIVATIONS_DIR / model_slug(model_name)
        if not (acts_dir / "harmful_train.pt").exists():
            raise FileNotFoundError(
                f"No cached activations at {acts_dir}. Run scripts/refusal_misaligned.py --model {model_name} first."
            )
        harmful_acts, harmless_acts = load_activations(acts_dir, "train")
        unit_directions, _ = compute_directions(harmful_acts, harmless_acts)
        return unit_directions[layer].float()

    direction_path = MODELS_DIR / model_slug(model_name) / (m2_dir or VARIANT_DIRS["M2"]) / "direction.pt"
    if not direction_path.exists():
        raise FileNotFoundError(
            f"No M2 steering vector found at {direction_path} (M2.3 reuses it). "
            f"Run scripts/refusal_misaligned.py --model {model_name} first."
        )
    return load_direction(direction_path)["direction"]


def load_variant(model_name: str, variant: str, device: str | None = None, alpha_override: float | None = None,
                  layer: int | None = None, m2_dir: str | None = None):
    """
    Returns (model, tokenizer, hook_handles). `hook_handles` is a (possibly
    empty) list of forward-hook handles the caller must pass to
    evaluations.common.remove_hooks(...) once finished, e.g.:

        model, tokenizer, handles = load_variant(model_name, variant)
        try:
            ... run benchmarks ...
        finally:
            remove_hooks(handles)

    `alpha_override`, if set, replaces the steering coefficient MAGNITUDE
    for the M1_risky+M2 / M1_bad_medical+M2 / M1_bad_medical+M2.1 /
    M1_medical_M2toward composites (normally abs(M2's own saved coef)) --
    lets you sweep how strongly the steering pushes without re-running M2
    itself. These are the only composites that steer TOWARDS refusal
    (positive coef); every composite that steers AWAY from refusal
    (M1_medical-M2, M1_medical+M2, M1_risky_M2away, M1_medical+M2.1,
    M1_bad_medical_M2away) does so via directional ablation instead, which
    alpha_override doesn't apply to. alpha_override should always be given
    as a positive magnitude. Ignored by every other variant.

    `layer`, if set, only affects M2.3: recomputes the unit refusal
    direction AT THAT DECODER LAYER from model_name's cached train
    activations (same mean-difference method refusal_misaligned.py used to
    pick M2's own layer), instead of reusing M2's saved direction -- lets
    you compare ablating a different layer's refusal representation. Still
    applied at every layer, same as the default path. Ignored by every
    other variant.

    `m2_dir`, if set, overrides which models/<slug>/ subfolder is read for
    M2's saved direction -- affects any variant that reuses M2's artifact
    (M1_medical-M2, M2.3, and the M1_*+M2 composites unless they
    reuse M2.1 instead). Default (None) is VARIANT_DIRS["M2"]
    (M2_steer_against_refusal). Lets you point these at an alternate saved
    M2 run (e.g. one saved via refusal_misaligned.py --out_suffix) without
    overwriting the canonical artifact. Ignored by variants that don't
    reuse M2's direction at all (e.g. M2.1, M2.2, base, M1_*).
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}, expected one of {VARIANTS}")

    device = device or get_device()
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    print(f"device={device}, cuda_available={torch.cuda.is_available()}, "
          f"model device={next(model.parameters()).device}")
    handles = []

    if variant == "base":
        pass
    elif variant in ("M1", "M1_risky_financial_advice", "M1_good_medical_advice", "M1_bad_medical_advice"):
        model = _merge_m1_adapter(model, model_name, variant, device)
    elif variant in _COMPOSITE_M1_BASE:
        model = _merge_m1_adapter(model, model_name, _COMPOSITE_M1_BASE[variant], device)

        direction_variant = _COMPOSITE_DIRECTION_VARIANT.get(variant, "M2")
        variant_dir = m2_dir if (m2_dir and direction_variant == "M2") else VARIANT_DIRS[direction_variant]
        direction_path = MODELS_DIR / model_slug(model_name) / variant_dir / "direction.pt"
        if not direction_path.exists():
            script = "refusal_misaligned_simple.py" if direction_variant == "M2.1" else "refusal_misaligned.py"
            raise FileNotFoundError(
                f"No {direction_variant} steering vector found at {direction_path} ({variant} reuses it). "
                f"Run scripts/{script} --model {model_name} first."
            )
        saved = load_direction(direction_path)

        if variant in ("M1_risky+M2", "M1_bad_medical+M2", "M1_bad_medical+M2.1", "M1_medical_M2toward"):
            # M2/M2.1 steer AWAY from refusal (their saved coef is negative);
            # this composite steers TOWARDS refusal on top of the finetune,
            # so flip the sign -- magnitude inferred from whatever the
            # underlying coef currently is, not hardcoded (unless
            # alpha_override is set, e.g. to sweep whether a stronger push
            # towards refusal further lowers attack success rate). Same
            # mechanism for risky_financial_advice/bad_medical_advice/
            # good_medical_advice M1 checkpoints, whether reusing M2's
            # direction or M2.1's (M1_bad_medical+M2.1).
            positive_coef = alpha_override if alpha_override is not None else abs(saved["coef"])
            handles = register_steering_hooks(model, saved["direction"], "additive", positive_coef, saved["layers"])
        else:  # M1_medical-M2 / M1_medical+M2 / M1_risky_M2away / M1_medical+M2.1 / M1_bad_medical_M2away
            # Steering AWAY from refusal is always interpreted as directional
            # ablation of the saved direction, at every layer via the dual
            # (pre-mixer + pre-MLP) ablation points -- same mechanism as
            # M2.3, applied to the M1 finetune instead of the base model.
            n_layers = len(get_decoder_layers(model))
            handles = register_dual_point_ablation_hooks(model, saved["direction"], list(range(n_layers)))
    elif variant == "M2.3":
        # Directional ablation, reusing M2's saved (raw, unnormalized)
        # direction -- renormalized to unit length here, since the ablation
        # projection formula h - (h.d)d only isolates exactly the
        # d-component when ||d||=1. Applied at both the pre-mixer and
        # pre-MLP points of every decoder layer (not just the single layer
        # M2 was tuned at, and not just once per layer) -- the refusal
        # direction is written into the residual stream at every layer, and
        # a single end-of-layer hook would let the mixer's own output
        # transiently reintroduce it before the MLP processes it.
        direction = _resolve_m2_ablation_direction(model_name, layer, m2_dir=m2_dir)
        n_layers = len(get_decoder_layers(model))
        handles = register_dual_point_ablation_hooks(model, direction, list(range(n_layers)))
    elif variant == "M2.2":
        # Directional ablation using M2.1/M2.2's OWN saved direction (from
        # refusal_misaligned_simple.py, distinct from M2's), applied at
        # every layer -- same mechanism/reasoning as M2.3 above, just reusing
        # the simple script's artifact instead of the full pipeline's.
        direction_path = MODELS_DIR / model_slug(model_name) / VARIANT_DIRS["M2.2"] / "direction.pt"
        if not direction_path.exists():
            raise FileNotFoundError(
                f"No M2.2 steering vector found at {direction_path}. "
                f"Run scripts/refusal_misaligned_simple.py --model {model_name} first."
            )
        direction = load_direction(direction_path)["direction"]
        n_layers = len(get_decoder_layers(model))
        handles = register_dual_point_ablation_hooks(model, direction, list(range(n_layers)))
    else:
        direction_path = MODELS_DIR / model_slug(model_name) / VARIANT_DIRS[variant] / "direction.pt"
        if not direction_path.exists():
            script = "refusal_misaligned_simple.py" if variant == "M2.1" else "refusal_misaligned.py"
            raise FileNotFoundError(
                f"No {variant} steering vector found at {direction_path}. "
                f"Run scripts/{script} --model {model_name} first."
            )
        saved = load_direction(direction_path)
        # alpha_override replaces the coefficient MAGNITUDE only, preserving
        # this variant's own sign (e.g. M2's saved coef is negative, steering
        # away from refusal) -- same convention as the M1+M2 composites above.
        coef = saved["coef"]
        if alpha_override is not None:
            coef = (1 if coef >= 0 else -1) * abs(alpha_override)
        handles = register_steering_hooks(model, saved["direction"], saved["mode"], coef, saved["layers"])

    model.eval()
    return model, tokenizer, handles


# ---------------------------------------------------------------------------
# Dual-point directional ablation (ported from mark2/AISafety's
# register_directional_ablation_hooks). This is the sole ablation mechanism
# load_variant uses for every ablation-based variant (M2.3, M1_medical-M2,
# M2.2) -- see register_dual_point_ablation_hooks below.
# ---------------------------------------------------------------------------

def _make_ablation_pre_hook(direction):
    """Forward-pre-hook form of directional ablation (Arditi et al., Eq. 4):
    x' = x - r^r^Tx. Unlike make_ablation_hook (a forward_hook on a whole
    decoder layer's *output*), this hooks a submodule's *input* so it can
    intercept the residual stream at a specific point inside the layer
    rather than only once at the very end."""
    d = direction.float()
    d = d / d.norm()

    def hook(_module, args):
        hidden = args[0]
        dv = d.to(hidden.dtype).to(hidden.device)
        proj = (hidden @ dv).unsqueeze(-1) * dv
        return (hidden - proj, *args[1:])

    return hook


def _get_ablation_points(layer):
    """Return (pre_mixer_module, pre_mlp_module) whose *inputs* are the two
    residual-stream ablation points from Arditi et al.: x^(l) (input to the
    layer's attention/token-mixer sub-block) and x~^(l) (input to its MLP
    sub-block, i.e. immediately after the mixer's output has been added back
    to the residual stream). Returns None if a layer doesn't expose both
    (caller should fall back to whole-layer output hooking for it)."""
    if hasattr(layer, "input_layernorm") and hasattr(layer, "post_attention_layernorm"):
        return layer.input_layernorm, layer.post_attention_layernorm
    return None


def register_dual_point_ablation_hooks(model, direction, layers):
    """True directional ablation (Arditi et al.), applied at *both*
    residual-stream points within each of the given decoder layers -- the
    input to the attention/token-mixer sub-block and the input to the MLP
    sub-block -- rather than only once at the end of the layer (as
    register_ablation_steering_hooks does). This closes the gap where the
    mixer's own output could transiently reintroduce `direction` into the
    stream before the MLP processes it. Falls back to a single end-of-layer
    hook (make_ablation_hook) for any layer that doesn't expose the expected
    norm submodule boundary."""
    all_layers = get_decoder_layers(model)
    pre_hook_fn = _make_ablation_pre_hook(direction)
    fallback_hook_fn = make_ablation_hook(direction)
    handles = []
    for i in layers:
        layer = all_layers[i]
        points = _get_ablation_points(layer)
        if points is None:
            handles.append(layer.register_forward_hook(fallback_hook_fn))
            continue
        pre_mixer_norm, pre_mlp_norm = points
        handles.append(pre_mixer_norm.register_forward_pre_hook(pre_hook_fn))
        handles.append(pre_mlp_norm.register_forward_pre_hook(pre_hook_fn))
    return handles


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
