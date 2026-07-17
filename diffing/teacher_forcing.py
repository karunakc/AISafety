"""
Shared plumbing for comparing two models' next-token distributions under
TEACHER FORCING: generate once with one model, then force both models over
the exact same token sequence so their per-position logits are directly
comparable (any difference is attributable to the models, not to them having
already produced different text).

Used by both diffing/method8_jsd.py (JSD/KL/TV/Hellinger between the two
logit tensors) and diffing/method9_refusal_probability.py (refusal-phrase
probability from the same logit tensors) -- both need the identical
(full_ids, logits_a, logits_b) triple, they just post-process it differently,
so the generation + forward-pass logic lives here once.
"""

import torch

Sequence = torch.Tensor  # [1, seq_len] LongTensor


def build_chat_prefix(tokenizer, prompt, device):
    """Chat-templates a single user turn and tokenizes it (no special tokens
    added twice -- add_generation_prompt=True already appends the
    assistant-turn opener). Returns [1, prefix_len] LongTensor on `device`.
    Shared by generate_and_teacher_force (below) and free_generation.py,
    which both need the identical prompt -> prefix-token-ids step but then
    diverge (shared vs. independent continuations)."""
    messages = [{"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)


def generate_and_teacher_force(model_a, model_b, tokenizer, prompt, max_new_tokens=64):
    """Greedily generate a continuation for `prompt` with model_a, then
    teacher-force BOTH models over the shared (prompt + continuation) token
    sequence.

    Returns a dict:
        full_ids    [1, prefix_len+gen_len] the complete token sequence, on
                    model_a's device
        logits_a    [1, gen_len, vocab] model_a's next-token logits, restricted
                    to the generated span (row i predicts full_ids[:, prefix_len+i])
        logits_b    [1, gen_len, vocab] same for model_b, moved to model_a's
                    device for direct comparison
        prefix_len  int, length of the tokenized prompt (chat-templated)
        gen_len     int, number of tokens model_a generated (0 if it emitted
                    EOS immediately -- callers should treat that as "nothing to
                    compare" for this prompt)

    Position bookkeeping: full_ids occupies indices [0, prefix_len+gen_len).
    The generated span is [prefix_len, prefix_len+gen_len). Position i's
    logits (from a causal LM) predict token i+1, so the logits that PREDICT
    the generated span start one index earlier than the span itself and stop
    one before the sequence end -- i.e. logits[:, prefix_len-1 : prefix_len-1+gen_len].
    That slicing is done here so callers never have to re-derive it.
    """
    prefix_ids = build_chat_prefix(tokenizer, prompt, model_a.device)
    prefix_len = prefix_ids.shape[1]

    with torch.no_grad():
        full_ids = model_a.generate(
            prefix_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    gen_len = full_ids.shape[1] - prefix_len
    if gen_len == 0:
        vocab = model_a.config.vocab_size
        empty = torch.empty(1, 0, vocab)
        return {
            "full_ids": full_ids, "logits_a": empty, "logits_b": empty,
            "prefix_len": prefix_len, "gen_len": 0,
        }

    with torch.no_grad():
        logits_a = model_a(full_ids).logits
        logits_b = model_b(full_ids.to(model_b.device)).logits.to(model_a.device)

    start, end = prefix_len - 1, full_ids.shape[1] - 1
    logits_a, logits_b = logits_a[:, start:end], logits_b[:, start:end]
    return {
        "full_ids": full_ids, "logits_a": logits_a, "logits_b": logits_b,
        "prefix_len": prefix_len, "gen_len": gen_len,
    }


__all__ = ["build_chat_prefix", "generate_and_teacher_force"]
