"""EQ-Bench and EmoBench -- the "EmoBench" column of the evaluation plan.

Both benchmarks ship a ground-truth answer, so we score against it directly
rather than going through a separate LLM-as-judge:

  - EQ-Bench: the model predicts 0-10 intensity scores for 4 candidate
    emotions; we diff them against the human reference and report a 0-100
    score (simplified re-implementation of the official difference-based
    metric -- not the exact original scoring script).
  - EmoBench: multiple-choice (emotion recognition, cause attribution, and
    action choice); scored via length-normalized log-likelihood over the
    answer choices (see evaluations.common.score_choices), reported as
    accuracy.
"""

import ast
import re

from datasets import load_dataset
from tqdm import tqdm

from eval_common import chat_generate, score_choices

EQ_BENCH_DATASET = dict(path="pbevan11/EQ-Bench", split="validation")
EMOBENCH_DATASET = dict(path="SahandSab/EmoBench", configs=["emotional_understanding", "emotional_application"])

EMOTION_LINE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z \-']*)\s*:\s*(-?\d+(?:\.\d+)?)", re.MULTILINE)


def _parse_emotion_scores(text):
    return {name.strip().lower(): float(score) for name, score in EMOTION_LINE_RE.findall(text)}


def _eq_bench_item_score(response, reference_fullscale):
    ref = ast.literal_eval(reference_fullscale)
    emotions = [ref[f"emotion{i}"] for i in range(1, 5)]
    ref_scores = [float(ref[f"emotion{i}_score"]) for i in range(1, 5)]

    predicted = _parse_emotion_scores(response)
    pred_scores = [predicted.get(e.lower(), 0.0) for e in emotions]

    diffs = [min(abs(p - r), 10.0) for p, r in zip(pred_scores, ref_scores)]
    return max(0.0, 10.0 - sum(diffs) / 4.0) * 10.0  # 0-100 scale


def run_eq_bench(model, tokenizer, n_prompts=50):
    ds = load_dataset(EQ_BENCH_DATASET["path"], split=EQ_BENCH_DATASET["split"])
    rows = ds.select(range(min(n_prompts, len(ds))))

    scores = []
    for row in tqdm(rows, desc="evaluating eq_bench"):
        # EQ-Bench answers are just 4 short "Emotion: score" lines -- the default
        # max_new_tokens=512 budget is ~5-10x oversized for that, and since generation
        # is autoregressive (one token at a time), the wasted budget directly multiplies
        # wall-clock time even with fast per-token decoding.
        response = chat_generate(model, tokenizer, row["prompt"], do_sample=False, max_new_tokens=128)
        try:
            scores.append(_eq_bench_item_score(response, row["reference_answer_fullscale"]))
        except (ValueError, SyntaxError, KeyError):
            continue

    return {"benchmark": "eq_bench", "n": len(scores), "mean_score": sum(scores) / len(scores) if scores else None}


def _emobench_understanding_prompt(row, field):
    choices = row[f"{field}_choices"]
    options = "\n".join(f"- {c}" for c in choices)
    return (
        f"Scenario: {row['scenario']}\n\n"
        f"Question: What {'emotion is' if field == 'emotion' else 'most likely caused'} "
        f"{row['subject']} {'feeling' if field == 'emotion' else 'this'}?\n"
        f"Options:\n{options}\n\nAnswer with exactly one of the options above."
    )


def _emobench_application_prompt(row):
    options = "\n".join(f"- {c}" for c in row["choices"])
    return (
        f"Scenario: {row['scenario']}\n\n"
        f"Question: What should {row['subject']} do?\n"
        f"Options:\n{options}\n\nAnswer with exactly one of the options above."
    )


def run_emobench(model, tokenizer, n_prompts=50):
    correct, total = 0, 0

    ds = load_dataset(EMOBENCH_DATASET["path"], "emotional_understanding", split="train")
    rows = ds.select(range(min(n_prompts, len(ds))))
    for row in tqdm(rows, desc="evaluating emobench (understanding)"):
        for field in ("emotion", "cause"):
            prompt = _emobench_understanding_prompt(row, field)
            choices = row[f"{field}_choices"]
            label = row[f"{field}_label"]
            best_idx, _ = score_choices(model, tokenizer, prompt, choices)
            correct += int(choices[best_idx] == label)
            total += 1

    ds = load_dataset(EMOBENCH_DATASET["path"], "emotional_application", split="train")
    rows = ds.select(range(min(n_prompts, len(ds))))
    for row in tqdm(rows, desc="evaluating emobench (application)"):
        prompt = _emobench_application_prompt(row)
        choices, label = row["choices"], row["label"]
        best_idx, _ = score_choices(model, tokenizer, prompt, choices)
        correct += int(choices[best_idx] == label)
        total += 1

    return {"benchmark": "emobench", "n": total, "accuracy": correct / total if total else None}


def run_emotion_benchmarks(model, tokenizer, benchmarks=None, n_prompts=50):
    benchmarks = benchmarks or ["eq_bench", "emobench"]
    results = {}
    if "eq_bench" in benchmarks:
        results["eq_bench"] = run_eq_bench(model, tokenizer, n_prompts=n_prompts)
    if "emobench" in benchmarks:
        results["emobench"] = run_emobench(model, tokenizer, n_prompts=n_prompts)
    return results
