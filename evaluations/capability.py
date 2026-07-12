"""Capability benchmarks: MMLU-Pro, GSM8K, BBH -- via lm-evaluation-harness.

The already-loaded `model` (LoRA-merged for M1, or with steering hooks
registered for M2.x/M3.x) is wrapped directly in lm_eval's HFLM so the same
forward hooks fire during evaluation as during normal generation.
"""

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

CAPABILITY_TASKS = ["mmlu_pro", "gsm8k", "bbh_cot_fewshot"]


def _group_total_docs(task_name):
    """mmlu_pro (and bbh_cot_fewshot) are GROUPS of many subject/category
    subtasks; lm_eval's own `limit` applies PER SUBTASK, not to the group's
    combined total, so a flat `limit=N` barely trims a group whose subtasks
    are mostly already smaller than N. This walks the loaded task tree to
    get the true combined doc count across every subtask, so callers can
    convert a desired GROUP-total cap into the fractional limit that
    achieves it (a fraction <1.0 makes lm_eval trim each subtask
    proportionally to its own size -- see evaluator_utils.get_sample_size)."""
    task_dict = TaskManager().load_task_or_group([task_name])

    def walk(td):
        total = 0
        for _, task in td.items():
            total += walk(task) if isinstance(task, dict) else len(task.eval_docs)
        return total

    return walk(task_dict)


def run_capability_benchmarks(model, tokenizer, tasks=None, batch_size=4, limit=None, mmlu_pro_total_limit=None):
    """`limit` (lm_eval's native per-subtask cap) applies to every task in
    `tasks` except mmlu_pro when `mmlu_pro_total_limit` is set -- that
    instead caps mmlu_pro's TOTAL example count summed across all 14 subject
    subtasks, by converting it to a fractional limit so every subtask is
    trimmed proportionally to its own size (see _group_total_docs)."""
    tasks = tasks or CAPABILITY_TASKS
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

    results = {}
    remaining = list(tasks)
    if "mmlu_pro" in remaining and mmlu_pro_total_limit is not None:
        remaining.remove("mmlu_pro")
        total_docs = _group_total_docs("mmlu_pro")
        frac = min(1.0, mmlu_pro_total_limit / total_docs)
        mmlu_results = lm_eval.simple_evaluate(model=lm, tasks=["mmlu_pro"], limit=frac)
        results.update(mmlu_results["results"])

    if remaining:
        other_results = lm_eval.simple_evaluate(model=lm, tasks=remaining, limit=limit)
        results.update(other_results["results"])

    return results
