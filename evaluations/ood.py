"""OOD prompts "from the EM paper" -- data/first_plot_questions.yaml.

Generates responses to the paper's curated out-of-distribution questions and
scores each one for coherence and alignment using the paper's own judge
rubric (judge_prompts.coherent / judge_prompts.aligned), graded by a local
LLM-as-judge.

Self-contained: does not import from eval_common.py or any other module in
evaluations/ -- this file can be run/deployed on its own.
"""

import re
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
QUESTIONS_PATH = DATA_DIR / "first_plot_questions.yaml"

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
    # enable_thinking=False: Qwen3-style tokenizers default to emitting a <think>...</think>
    # block before the answer; these evals want the direct response, not the reasoning trace,
    # unless the caller explicitly asks for it. Tokenizers without this template feature just
    # ignore the unrecognized kwarg.
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


_judge_cache = {}


def _load_judge(judge_model_name, device):
    if judge_model_name not in _judge_cache:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(judge_model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(judge_model_name).to(device)
        model.eval()
        _judge_cache[judge_model_name] = (model, tokenizer)
    return _judge_cache[judge_model_name]


def _judge_score(judge_model, judge_tokenizer, judge_prompt, device):
    # judge_prompt comes straight from first_plot_questions.yaml's judge_prompts.aligned /
    # judge_prompts.coherent (see load_questions below) -- it's already a complete rubric
    # (scale, edge cases like CODE/REFUSAL, and "respond with just the number"), so it's
    # sent as-is with no extra system-prompt wrapper.
    messages = [{"role": "user", "content": judge_prompt}]
    inputs = judge_tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, truncation=True, add_generation_prompt=True
    ).to(device)
    with torch.no_grad():
        output_ids = judge_model.generate(
            **inputs, max_new_tokens=20, do_sample=False, pad_token_id=judge_tokenizer.eos_token_id
        )
    prompt_len = inputs["input_ids"].shape[1]
    # Keep the raw judge text as-is (the judge is asked to emit only the number, but
    # occasionally wraps it in stray text) alongside a best-effort numeric parse --
    # the raw text is what gets persisted in records; the parsed float is what
    # aggregation below actually averages.
    text = judge_tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
    match = re.search(r"\d+", text)
    score = float(match.group(0)) if match else None
    return text, score


def load_questions(n_groups=8):
    with open(QUESTIONS_PATH) as f:
        questions = yaml.safe_load(f)
    judge_prompts = questions[0]["judge_prompts"]
    eval_q = []
    for item in questions[:n_groups]:
        eval_q.extend(item["paraphrases"])
    return eval_q, judge_prompts


def run_ood_benchmark(
    model, tokenizer, judge_model_name="Qwen/Qwen3-4B-Instruct-2507", device="cuda",
    n_groups=8, n_samples=10, enable_thinking=False,
):
    """For each question, sample `n_samples` independent responses (chat_generate
    defaults to do_sample=True, so repeated calls genuinely differ) and average
    their coherence/alignment judge scores into one per-question score, before
    averaging those per-question scores into the final benchmark result. This
    smooths out generation-sampling noise in the per-question estimate."""
    eval_q, judge_prompts = load_questions(n_groups)
    judge_model, judge_tokenizer = _load_judge(judge_model_name, device)

    records = []
    coherence_scores, alignment_scores = [], []
    for question in tqdm(eval_q, desc="evaluating OOD prompts"):
        question_coherence, question_alignment = [], []
        for iteration in range(n_samples):
            raw_response = chat_generate(model, tokenizer, question, enable_thinking=enable_thinking)
            thinking, response = _split_thinking(raw_response)

            # Judge scores the final answer only, not the reasoning trace.
            coherence_prompt = judge_prompts["coherent"].format(question=question, answer=response)
            coherence_raw, coherence_score = _judge_score(judge_model, judge_tokenizer, coherence_prompt, device)
            if coherence_score is not None:
                question_coherence.append(coherence_score)

            alignment_prompt = judge_prompts["aligned"].format(question=question, answer=response)
            alignment_raw, alignment_score = _judge_score(judge_model, judge_tokenizer, alignment_prompt, device)
            if alignment_score is not None:
                question_alignment.append(alignment_score)

            records.append({
                "question": question,
                "iteration": iteration,
                "response": response,
                "thinking": thinking,
                "coherence_score": coherence_score,
                "coherence_raw": coherence_raw,
                "alignment_score": alignment_score,
                "alignment_raw": alignment_raw,
            })

        if question_coherence:
            coherence_scores.append(sum(question_coherence) / len(question_coherence))
        if question_alignment:
            alignment_scores.append(sum(question_alignment) / len(question_alignment))

    return {
        "benchmark": "ood_em_paper",
        "n": len(eval_q),
        "n_samples_per_question": n_samples,
        "thinking_enabled": enable_thinking,
        "mean_coherence": sum(coherence_scores) / len(coherence_scores) if coherence_scores else None,
        "mean_alignment": sum(alignment_scores) / len(alignment_scores) if alignment_scores else None,
        "records": records,
    }
