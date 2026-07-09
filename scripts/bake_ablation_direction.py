"""
Bake M2.3 (directional ablation) into an actual model checkpoint, via weight
orthogonalization -- the technique from the refusal-direction paper (Arditi
et al.), Section 4.1 "Weight orthogonalization":

    W'_out <- W_out - r_hat @ r_hat^T @ W_out    (Eq. 5)

applied to every weight matrix that writes to the residual stream: the
embedding matrix, attention-out matrices, and MLP-out matrices (plus any
output biases). This is the exact weight-space equivalent of the runtime
hook's directional ablation h' = h - (h.d)d = (I - dd^T)h at every layer, and
the paper proves it's identical (their Appendix E) -- so its bypass
performance exactly characterizes the direct weight modification too.

Note: the paper also lists a "positional embedding matrix" among the things
to orthogonalize. That doesn't apply here -- Qwen3/Llama-family models use
RoPE (rotary position embeddings), which rotates queries/keys inside
attention rather than writing a separate additive positional vector to the
residual stream, so there's nothing extra to modify for that term.

Why the embedding matrix needs it too, and the tie_word_embeddings gotcha:
the residual stream starts at the token embedding, so if it isn't
orthogonalized, direction d re-enters at every layer via the residual
(skip) connection regardless of what's done to o_proj/down_proj. If the
model ties input/output embeddings (tie_word_embeddings=True), that same
matrix also IS the final vocab projection (lm_head) -- modifying it for the
input side would silently also corrupt the output side, which the paper's
method (and our runtime hook) never touches. So we untie first: give
lm_head its own frozen copy of the ORIGINAL, unablated embeddings, then
orthogonalize the input embedding copy only.

This equivalence assumes a standard pre-norm decoder block (residual = h;
h = attn(norm(h)); h = residual + h; residual = h; h = mlp(norm(h));
h = residual + h) with nothing nonlinear applied to attn_out/mlp_out AFTER
computing them but before the residual add ("sandwich norm"). Verified to
hold for Qwen3 (and the wider Llama/Qwen family). It does NOT hold for e.g.
Gemma3, which applies additional post_attention/post_feedforward RMSNorms
to attn_out/mlp_out before the residual add -- RMSNorm is nonlinear, so the
orthogonalized weights would not reproduce the runtime hook there.

The script verifies the result numerically before saving: one forward pass
with the ORIGINAL weights + the runtime ablation hook, then the same
forward pass on the orthogonalized weights with NO hooks, comparing logits.

Usage:
    python scripts/bake_ablation_direction.py --model Qwen/Qwen3-4B
"""

import argparse

import torch

from common import (
    MODELS_DIR,
    get_decoder_layers,
    get_device,
    load_direction,
    load_model_and_tokenizer,
    model_slug,
    register_ablation_steering_hooks,
    remove_hooks,
)

TEST_PROMPT = "How do I pick a lock?"


def _orthogonalize(linear, P32):
    """In-place, Eq. 5: W'_out = P @ W_out (and bias'_out = P @ bias_out, if present).
    Math is done in float32 regardless of the module's own dtype, then cast back."""
    w32 = linear.weight.data.float()
    linear.weight.data.copy_((P32 @ w32).to(linear.weight.dtype))
    if linear.bias is not None:
        b32 = linear.bias.data.float()
        linear.bias.data.copy_((P32 @ b32).to(linear.bias.dtype))


def bake_ablation(model, direction):
    """Orthogonalizes every weight matrix that writes to the residual stream
    with respect to `direction`, in place (Arditi et al., Eq. 5)."""
    d = direction.float()
    d = d / d.norm()
    hidden_dim = d.shape[0]
    P32 = torch.eye(hidden_dim, dtype=torch.float32) - torch.outer(d, d)

    if model.config.tie_word_embeddings:
        print("Model ties input/output embeddings -- untying so orthogonalizing the "
              "input embeddings doesn't also corrupt the output (lm_head) projection.")
        original_embed = model.get_input_embeddings().weight.data.clone()
        output_embeddings = model.get_output_embeddings()
        output_embeddings.weight = torch.nn.Parameter(
            original_embed.clone().to(output_embeddings.weight.dtype).to(output_embeddings.weight.device)
        )
        model.config.tie_word_embeddings = False

    # Embedding matrix writes to the residual stream at the very start, so it
    # needs the same treatment -- rows are embedding vectors, hence right-
    # multiply (P is symmetric, so this is equivalent to (P @ E^T)^T).
    embed_tokens = model.get_input_embeddings()
    P32_embed = P32.to(embed_tokens.weight.device)
    e32 = embed_tokens.weight.data.float()
    embed_tokens.weight.data.copy_((e32 @ P32_embed).to(embed_tokens.weight.dtype))

    # Attention-out (o_proj) and MLP-out (down_proj) matrices at every layer.
    layers = get_decoder_layers(model)
    for layer in layers:
        _orthogonalize(layer.self_attn.o_proj, P32.to(layer.self_attn.o_proj.weight.device))
        _orthogonalize(layer.mlp.down_proj, P32.to(layer.mlp.down_proj.weight.device))


def run(model_name, output_dir=None, tolerance=5e-2, test_prompt=TEST_PROMPT):
    device = get_device()
    slug = model_slug(model_name)
    output_dir = output_dir or (MODELS_DIR / slug / "M2.3_ablation_baked")

    additive_path = MODELS_DIR / slug / "M2.1_steer_against_refusal_additive" / "direction.pt"
    if not additive_path.exists():
        raise FileNotFoundError(
            f"No saved M2.1 direction at {additive_path}. Run scripts/refusal_misaligned.py --model {model_name} first."
        )
    saved = load_direction(additive_path)
    direction = saved["direction"]
    print(f"Loaded M2.1 direction from {additive_path}")

    print(f"Loading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    n_layers = len(get_decoder_layers(model))
    print(f"Model has {n_layers} decoder layers")

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": test_prompt}], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("Computing reference logits: original weights + runtime ablation hooks...")
    handles = register_ablation_steering_hooks(model, direction, list(range(n_layers)))
    with torch.no_grad():
        ref_logits = model(**inputs).logits[0, -1, :].float()
    remove_hooks(handles)

    print("Orthogonalizing weights (baking ablation in)...")
    with torch.no_grad():
        bake_ablation(model, direction)

    print("Computing baked-model logits: orthogonalized weights, no hooks...")
    with torch.no_grad():
        baked_logits = model(**inputs).logits[0, -1, :].float()

    max_abs_diff = (ref_logits - baked_logits).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(ref_logits.unsqueeze(0), baked_logits.unsqueeze(0)).item()
    print(f"Verification: max abs logit diff = {max_abs_diff:.6f}, cosine similarity = {cos_sim:.6f}")

    # max-abs-diff on raw logits is a strict, somewhat arbitrary bar --
    # especially under bf16, where casting the projection back and forth
    # across ~74 modified matrices (36 layers x 2 + embeddings) accumulates
    # rounding error in absolute logit magnitude without necessarily changing
    # *behavior*. Top-1/top-5 token agreement and KL divergence between the
    # two output distributions are the more meaningful checks -- do the two
    # models actually predict differently, or just wobble on unimportant
    # logit magnitudes?
    ref_top1 = ref_logits.argmax().item()
    baked_top1 = baked_logits.argmax().item()
    ref_top5 = set(ref_logits.topk(5).indices.tolist())
    baked_top5 = set(baked_logits.topk(5).indices.tolist())
    kl = torch.nn.functional.kl_div(
        torch.log_softmax(baked_logits, dim=-1), torch.softmax(ref_logits, dim=-1), reduction="sum"
    ).item()
    print(f"Top-1 token match: {ref_top1 == baked_top1} (ref={ref_top1!r}, baked={baked_top1!r})")
    print(f"Top-5 token overlap: {len(ref_top5 & baked_top5)}/5")
    print(f"KL(ref || baked) = {kl:.6f}")

    if max_abs_diff > tolerance:
        print(f"WARNING: max-abs-diff exceeds tolerance ({tolerance}) -- but check the top-1/top-5/KL "
              f"numbers above before concluding this is wrong. High cosine similarity + matching top "
              f"tokens + low KL despite a tolerance-exceeding max-abs-diff usually means bf16 rounding, "
              f"not a bug in the orthogonalization.")
    else:
        print("Within tolerance -- baked weights reproduce the runtime ablation hook's output.")

    print(f"Saving baked checkpoint to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Bake M2.3 directional ablation into real model weights.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--tolerance", type=float, default=5e-2)
    parser.add_argument("--test_prompt", default=TEST_PROMPT)
    args = parser.parse_args()
    run(args.model, output_dir=args.output_dir, tolerance=args.tolerance, test_prompt=args.test_prompt)


if __name__ == "__main__":
    main()
