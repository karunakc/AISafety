"""EQ-Bench and EmoBench -- the "EmoBench" column of the evaluation plan.

Both benchmarks ship a ground-truth answer, so we score against it directly
rather than going through a separate LLM-as-judge:

  - EQ-Bench: the model predicts 0-10 intensity scores for 4 candidate
    emotions; we diff them against the human reference and report a 0-100
    score (simplified re-implementation of the official difference-based
    metric -- not the exact original scoring script).
  - EmoBench: multiple-choice (emotion recognition, cause attribution, and
    action choice); scored via length-normalized log-likelihood over the
    answer choices (see score_choices below), reported as accuracy.

Self-contained: does not import from eval_common.py or any other module in
evaluations/ -- this file can be run/deployed on its own.
"""

import ast
import re

import torch
from datasets import load_dataset
from tqdm import tqdm

EQ_BENCH_DATASET = dict(path="pbevan11/EQ-Bench", split="validation")
EMOBENCH_DATASET = dict(path="SahandSab/EmoBench", configs=["emotional_understanding", "emotional_application"])

EMOTION_LINE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z \-']*)\s*:\s*(-?\d+(?:\.\d+)?)", re.MULTILINE)

THINK_BLOCK_RE = re.compile(r"^\s*<think>(.*?)</think>\s*", re.DOTALL)


def _split_thinking(text):
    """Split a raw generation into (thinking, response). Only emitted when
    enable_thinking=True (Qwen3-style tokenizers then have the model open with
    a <think>...</think> block); otherwise thinking is None and response is
    the text unchanged."""
    match = THINK_BLOCK_RE.match(text)
    if not match:
        return None, text
    return match.group(1).strip(), text[match.end():].strip()


def chat_generate(
    model,
    tokenizer,
    prompt,
    system_prompt=None,
    max_new_tokens=512,
    do_sample=True,
    temperature=1.0,
    top_p=0.9,
    enable_thinking=False,
):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def score_choices(model, tokenizer, prompt, choices, system_prompt=None, enable_thinking=False):
    """
    Length-normalized log-likelihood multiple-choice scoring: returns the
    index of the choice whose text has the highest average per-token
    log-probability when appended as the assistant's response to `prompt`.
    Avoids brittle text parsing for benchmarks with a fixed answer set
    (e.g. EmoBench).

    enable_thinking=True is accepted for interface consistency with the rest
    of the eval suite, but scoring appends `choice` directly after the prompt
    with no <think> block in between -- if the tokenizer's template expects
    thinking first, this will systematically underscore every choice rather
    than meaningfully account for a reasoning trace. Leave this False unless
    you're specifically investigating that effect.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    prefix = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
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


def run_eq_bench(model, tokenizer, n_prompts=50, enable_thinking=False):
    ds = load_dataset(EQ_BENCH_DATASET["path"], split=EQ_BENCH_DATASET["split"])
    rows = ds.select(range(min(n_prompts, len(ds))))

    scores = []
    records = []
    for row in tqdm(rows, desc="evaluating eq_bench"):
        # EQ-Bench answers are just 4 short "Emotion: score" lines -- 128 tokens is
        # already ~5-10x oversized for that, and since generation is autoregressive
        # (one token at a time), a bigger budget directly multiplies wall-clock time
        # even with fast per-token decoding. With thinking enabled the reasoning trace
        # itself needs real headroom, so give it the same 2048 budget as safety.py.
        max_new_tokens = 2048 if enable_thinking else 128
        raw_response = chat_generate(
            model, tokenizer, row["prompt"], do_sample=False, max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
        )
        thinking, response = _split_thinking(raw_response)
        try:
            score = _eq_bench_item_score(response, row["reference_answer_fullscale"])
        except (ValueError, SyntaxError, KeyError):
            continue
        scores.append(score)
        records.append({"prompt": row["prompt"], "response": response, "thinking": thinking, "score": score})

    return {
        "benchmark": "eq_bench",
        "n": len(scores),
        "thinking_enabled": enable_thinking,
        "mean_score": sum(scores) / len(scores) if scores else None,
        "records": records,
    }


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


def run_emobench(model, tokenizer, n_prompts=50, enable_thinking=False):
    correct, total = 0, 0

    ds = load_dataset(EMOBENCH_DATASET["path"], "emotional_understanding", split="train")
    rows = ds.select(range(min(n_prompts, len(ds))))
    for row in tqdm(rows, desc="evaluating emobench (understanding)"):
        for field in ("emotion", "cause"):
            prompt = _emobench_understanding_prompt(row, field)
            choices = row[f"{field}_choices"]
            label = row[f"{field}_label"]
            best_idx, _ = score_choices(model, tokenizer, prompt, choices, enable_thinking=enable_thinking)
            correct += int(choices[best_idx] == label)
            total += 1

    ds = load_dataset(EMOBENCH_DATASET["path"], "emotional_application", split="train")
    rows = ds.select(range(min(n_prompts, len(ds))))
    for row in tqdm(rows, desc="evaluating emobench (application)"):
        prompt = _emobench_application_prompt(row)
        choices, label = row["choices"], row["label"]
        best_idx, _ = score_choices(model, tokenizer, prompt, choices, enable_thinking=enable_thinking)
        correct += int(choices[best_idx] == label)
        total += 1

    return {
        "benchmark": "emobench",
        "n": total,
        "thinking_enabled": enable_thinking,
        "accuracy": correct / total if total else None,
    }


def run_emotion_benchmarks(model, tokenizer, benchmarks=None, n_prompts=50, enable_thinking=False):
    benchmarks = benchmarks or ["eq_bench", "emobench"]
    results = {}
    if "eq_bench" in benchmarks:
        results["eq_bench"] = run_eq_bench(model, tokenizer, n_prompts=n_prompts, enable_thinking=enable_thinking)
    if "emobench" in benchmarks:
        results["emobench"] = run_emobench(model, tokenizer, n_prompts=n_prompts, enable_thinking=enable_thinking)
    return results
