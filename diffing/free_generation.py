"""
FREE (unconstrained) generation comparison between two models -- the
counterpart to method8_jsd.py's teacher-forced comparison.

Under teacher forcing, both models see the SAME token history at every step,
so any difference in their next-token distributions is attributable to the
models themselves. Under free generation, each model conditions on its OWN
previously generated tokens -- the moment the two token streams diverge,
their contexts differ too, so a per-position distributional comparison
(JSD, top-k overlap, ...) stops being an apples-to-apples "same context,
different model" comparison and starts partly reflecting "different
context". This module still computes those comparisons at every position
(they're informative -- e.g. "how similar are the plausible next steps from
here" is a fair question even off-context), but call sites must report the
`diverged_at` position alongside any curve so readers know where the
comparison's meaning shifts. See `divergence_position`.

Metric computation only -- no plotting (see diffing/plots.py).
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def free_generate(model, tokenizer, prefix_ids, max_new_tokens=64, do_sample=False,
                   temperature=1.0, top_p=1.0, seed=None):
    """Generates a continuation from `prefix_ids` using the model's own
    `generate` (KV-cached internally -- no manual re-forward per step) with
    `output_scores=True`, so we get the EXACT pre-sampling next-token logits
    the model used to produce each token, for free (no separate re-scoring
    pass needed, which matters most under sampling: a fresh forward pass
    conditioned on the same prefix reproduces the same logits deterministically
    at eval time, but reusing `generate`'s own scores avoids relying on that
    and is simply cheaper).

    Returns a dict:
        token_ids  [gen_len] LongTensor, the generated continuation only
        text       str, decoded continuation (special tokens stripped)
        logits     [gen_len, vocab] float tensor, one row per generated step
        probs      [gen_len, vocab] float tensor, softmax(logits)
    All tensors are moved to CPU (these are collected across many prompts and
    two models; keeping them on GPU would exhaust memory fast).
    """
    if seed is not None:
        torch.manual_seed(seed)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens, do_sample=do_sample, pad_token_id=tokenizer.eos_token_id,
        output_scores=True, return_dict_in_generate=True,
    )
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)

    out = model.generate(prefix_ids, **gen_kwargs)
    gen_ids = out.sequences[:, prefix_ids.shape[1]:]
    logits = torch.stack(out.scores, dim=1)[0] if out.scores else prefix_ids.new_empty(0, model.config.vocab_size).float()
    probs = F.softmax(logits.float(), dim=-1)
    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return {
        "token_ids": gen_ids[0].cpu(),
        "text": text,
        "logits": logits.cpu(),
        "probs": probs.cpu(),
    }


def divergence_position(token_ids_a, token_ids_b):
    """Compares two independently-generated token-id sequences and returns
    (common_prefix_len, diverged):
        common_prefix_len  number of leading tokens at which both sequences
                            agree exactly (== the index of the first mismatch)
        diverged            True if a mismatch was found; False if one
                            sequence is a prefix of the other (e.g. the
                            shorter one hit EOS and stopped -- there's no
                            "differing token", just a differing length)
    `common_prefix_len` IS the position at which downstream JSD/top-k-overlap
    values stop being a same-context comparison (see module docstring)."""
    token_ids_a = token_ids_a.tolist() if torch.is_tensor(token_ids_a) else list(token_ids_a)
    token_ids_b = token_ids_b.tolist() if torch.is_tensor(token_ids_b) else list(token_ids_b)
    n = min(len(token_ids_a), len(token_ids_b))
    for i in range(n):
        if token_ids_a[i] != token_ids_b[i]:
            return i, True
    return n, False


def topk_overlap(logits_a, logits_b, k=10):
    """Fraction of each distribution's top-`k` token ids that the other
    distribution also ranks in its top-`k`, at every position: |topk_a ∩
    topk_b| / k, in [0, 1]. `logits_a`/`logits_b`: [seq_len, vocab] -- same
    seq_len (e.g. both truncated to min(len_a, len_b) by the caller).

    Interpretation: JSD says the two distributions differ in mass; top-k
    overlap says whether they differ in WHICH tokens they even consider --
    a high JSD with high top-k overlap means the models agree on the
    candidate continuations but reweight them differently (consistent with
    steering that suppresses/boosts existing candidates); a high JSD with
    LOW top-k overlap means the models are proposing genuinely different
    continuations, not just re-ranking the same ones.
    """
    topk_a = logits_a.topk(min(k, logits_a.shape[-1]), dim=-1).indices
    topk_b = logits_b.topk(min(k, logits_b.shape[-1]), dim=-1).indices
    overlaps = []
    for i in range(logits_a.shape[0]):
        set_a, set_b = set(topk_a[i].tolist()), set(topk_b[i].tolist())
        overlaps.append(len(set_a & set_b) / k)
    return torch.tensor(overlaps)


__all__ = ["free_generate", "divergence_position", "topk_overlap"]
