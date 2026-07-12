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
  - M2.3          -> directional ablation, applied at every layer, reusing
                     M2's saved (raw) direction renormalized to unit length
                     -- no separate artifact is written for it
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
                     direction added back at its SINGLE tuned layer with
                     M2's own (negative) sign preserved -- the SAME additive
                     steer-away-from-refusal mechanism M2 itself uses on the
                     base model, just applied on top of this finetune
                     instead of ablation. A more direct comparison against
                     M2's own mechanism than M1_medical-M2's ablation is.
  - M1_risky_M2away -> M1_risky_financial_advice's LoRA merged, then M2's
                     direction added back with M2's own (negative) sign
                     preserved -- same mechanism as M1_medical+M2, just on
                     the risky_financial_advice finetune instead. This is
                     the direct base-vector TRANSFER TEST: does the refusal
                     direction extracted on the base model still suppress
                     refusal, unchanged, after LoRA fine-tuning? (Distinct
                     from M1_risky+M2, which flips the sign to push TOWARDS
                     refusal as a counter-misalignment probe, not a
                     transfer test.)
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
    model_slug,
    register_ablation_steering_hooks,
    register_steering_hooks,
    remove_hooks,
)

VARIANTS = [
    "base", "M1", "M1_risky_financial_advice", "M1_good_medical_advice", "M2", "M2.3",
    "M1_risky+M2", "M1_medical-M2", "M1_medical+M2", "M1_risky_M2away",
]

VARIANT_DIRS = {
    "M1": "M1_emergent_misalignment",
    "M1_risky_financial_advice": "M1_risky_financial_advice",
    "M1_good_medical_advice": "M1_good_medical_advice",
    "M2": "M2_steer_against_refusal",
    # M2.3 / M1_risky+M2 / M1_medical-M2 have no artifact of their own --
    # they're composed at load time from the M1_*/M2 artifacts above.
}

# Which M1 checkpoint each M1+M2 composite variant is built on.
_COMPOSITE_M1_BASE = {
    "M1_risky+M2": "M1_risky_financial_advice",
    "M1_medical-M2": "M1_good_medical_advice",
    "M1_medical+M2": "M1_good_medical_advice",
    "M1_risky_M2away": "M1_risky_financial_advice",
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


def load_variant(model_name: str, variant: str, device: str | None = None, alpha_override: float | None = None):
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
    for the M1_risky+M2 / M1_medical+M2 composites (normally abs(M2's own
    saved coef)) -- lets you sweep how strongly the steering pushes without
    re-running M2 itself. Each composite applies its own sign (positive =
    towards refusal for M1_risky+M2, negative = away from refusal for
    M1_medical+M2); alpha_override should always be given as a positive
    magnitude. Ignored by every other variant.
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
    elif variant in ("M1", "M1_risky_financial_advice", "M1_good_medical_advice"):
        model = _merge_m1_adapter(model, model_name, variant, device)
    elif variant in _COMPOSITE_M1_BASE:
        model = _merge_m1_adapter(model, model_name, _COMPOSITE_M1_BASE[variant], device)

        direction_path = MODELS_DIR / model_slug(model_name) / VARIANT_DIRS["M2"] / "direction.pt"
        if not direction_path.exists():
            raise FileNotFoundError(
                f"No M2 steering vector found at {direction_path} ({variant} reuses it). "
                f"Run scripts/refusal_misaligned.py --model {model_name} first."
            )
        saved = load_direction(direction_path)

        if variant == "M1_risky+M2":
            # M2 steers AWAY from refusal (its saved coef is negative); this
            # composite steers TOWARDS refusal on top of the emergent
            # misalignment finetune, so flip the sign -- magnitude inferred
            # from whatever M2's own coef currently is, not hardcoded (unless
            # alpha_override is set, e.g. to sweep whether a stronger push
            # towards refusal further lowers attack success rate).
            positive_coef = alpha_override if alpha_override is not None else abs(saved["coef"])
            handles = register_steering_hooks(model, saved["direction"], "additive", positive_coef, saved["layers"])
        elif variant in ("M1_medical+M2", "M1_risky_M2away"):
            # SAME mechanism M2 itself uses on the base model (additive,
            # sign preserved -- steers AWAY from refusal), just applied on
            # top of an M1 finetune instead of ablating. For M1_medical+M2
            # this is a comparison against M1_medical-M2's ablation; for
            # M1_risky_M2away this IS the base-vector transfer test (does
            # the base model's refusal direction, unmodified, still
            # suppress refusal after LoRA fine-tuning?).
            negative_coef = -abs(alpha_override) if alpha_override is not None else saved["coef"]
            handles = register_steering_hooks(model, saved["direction"], "additive", negative_coef, saved["layers"])
        else:  # M1_medical-M2
            # Ablate M2's direction at every layer -- same mechanism as
            # M2.3, applied to the good_medical_advice finetune instead of
            # the base model.
            n_layers = len(get_decoder_layers(model))
            handles = register_ablation_steering_hooks(model, saved["direction"], list(range(n_layers)))
    elif variant == "M2.3":
        # Directional ablation, reusing M2's saved (raw, unnormalized)
        # direction -- renormalized to unit length here, since the ablation
        # projection formula h - (h.d)d only isolates exactly the
        # d-component when ||d||=1. Applied at every layer (not just the
        # single layer M2 was tuned at), since the refusal direction is
        # written into the residual stream at every layer and ablating only
        # one layer would let downstream layers reintroduce it.
        direction_path = MODELS_DIR / model_slug(model_name) / VARIANT_DIRS["M2"] / "direction.pt"
        if not direction_path.exists():
            raise FileNotFoundError(
                f"No M2 steering vector found at {direction_path} (M2.3 reuses it). "
                f"Run scripts/refusal_misaligned.py --model {model_name} first."
            )
        saved = load_direction(direction_path)
        n_layers = len(get_decoder_layers(model))
        handles = register_ablation_steering_hooks(model, saved["direction"], list(range(n_layers)))
    else:
        direction_path = MODELS_DIR / model_slug(model_name) / VARIANT_DIRS[variant] / "direction.pt"
        if not direction_path.exists():
            raise FileNotFoundError(
                f"No {variant} steering vector found at {direction_path}. "
                f"Run scripts/refusal_misaligned.py --model {model_name} first."
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
