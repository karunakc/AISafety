"""Capability benchmarks: MMLU-Pro, GSM8K, BBH -- via lm-evaluation-harness.

The already-loaded `model` (LoRA-merged for M1, or with steering hooks
registered for M2.x/M3.x) is wrapped directly in lm_eval's HFLM so the same
forward hooks fire during evaluation as during normal generation.
"""

import lm_eval
from lm_eval.models.huggingface import HFLM

CAPABILITY_TASKS = ["mmlu_pro"]#, "gsm8k", "bbh_cot_fewshot"]


MMLU_PRO_SUBTASKS = 14  # mmlu_pro expands into 14 per-subject tasks (biology, business, ...)


def run_capability_benchmarks(model, tokenizer, tasks=None, batch_size="auto", max_length=4096, limit=None, mmlu_pro_limit=None):
    # Without an explicit cap, HFLM sizes its batch-size probing (and KV-cache allocation)
    # off the model's configured context length. This model's native window is 262,144
    # tokens, so an uncapped probe tries to allocate KV-cache for that length and OOMs
    # even at batch_size=1 -- MMLU-Pro/GSM8K/BBH prompts are only a few thousand tokens
    # at most, so cap generously above that instead of trusting the model's native max.
    tasks = tasks or CAPABILITY_TASKS
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, max_length=max_length)

    results = {}
    if "mmlu_pro" in tasks and mmlu_pro_limit is not None:
        # simple_evaluate's `limit` applies per task, and mmlu_pro expands into 14
        # independent per-subject subtasks -- a single `limit` would apply to *each*
        # subject (14x the requested total), not to mmlu_pro as a whole. Split it into
        # its own call with a per-subject share of mmlu_pro_limit so the combined total
        # across all 14 subjects approximates the requested count (uneven subject sizes
        # mean it won't be exact).
        per_subject_limit = max(1, mmlu_pro_limit // MMLU_PRO_SUBTASKS)
        mmlu_pro_results = lm_eval.simple_evaluate(model=lm, tasks=["mmlu_pro"], limit=per_subject_limit)
        results.update(mmlu_pro_results["results"])
        tasks = [t for t in tasks if t != "mmlu_pro"]

    if tasks:
        remaining_results = lm_eval.simple_evaluate(model=lm, tasks=tasks, limit=limit)
        results.update(remaining_results["results"])

    return results
