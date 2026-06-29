"""Capability benchmarks: MMLU-Pro, GSM8K, BBH -- via lm-evaluation-harness.

The already-loaded `model` (LoRA-merged for M1, or with steering hooks
registered for M2.x/M3.x) is wrapped directly in lm_eval's HFLM so the same
forward hooks fire during evaluation as during normal generation.
"""

import lm_eval
from lm_eval.models.huggingface import HFLM

CAPABILITY_TASKS = ["mmlu_pro", "gsm8k", "bbh_cot_fewshot"]


def run_capability_benchmarks(model, tokenizer, tasks=None, batch_size=4, limit=None):
    tasks = tasks or CAPABILITY_TASKS
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    results = lm_eval.simple_evaluate(model=lm, tasks=tasks, limit=limit)
    return {task: results["results"][task] for task in results["results"]}
