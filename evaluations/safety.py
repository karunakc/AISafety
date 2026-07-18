"""Safety benchmarks: HarmBench, AdvBench, AgentHarm -- judged with judgezoo's
StrongREJECT judge (qylu4156/strongreject-15k-v1).

AgentHarm prompts describe agentic tool-use tasks; we evaluate the text-only
behaviour (no tool-execution sandbox) by generating a direct response and
scoring its harmfulness the same way as the other two benchmarks.

Self-contained: does not import from eval_common.py or any other module in
evaluations/ -- this file can be run/deployed on its own.
"""

import re

import torch
from datasets import load_dataset
from tqdm import tqdm

SAFETY_DATASETS = {
    # "harmbench": dict(path="walledai/HarmBench", name="standard", split="train", field="prompt"),
    "advbench": dict(path="walledai/AdvBench", name=None, split="train", field="prompt"),
    "agentharm": dict(path="ai-safety-institute/AgentHarm", name="harmful", split="test_public", field="prompt"),
}

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


def _load_judge():
    if "strong_reject" not in _judge_cache:
        from judgezoo import Judge

        _judge_cache["strong_reject"] = Judge.from_name("strong_reject")
    return _judge_cache["strong_reject"]


def _load_prompts(benchmark, n_prompts):
    cfg = SAFETY_DATASETS[benchmark]
    ds = load_dataset(cfg["path"], cfg["name"], split=cfg["split"]) if cfg["name"] else load_dataset(cfg["path"], split=cfg["split"])
    return list(ds[cfg["field"]][:n_prompts])


def run_safety_benchmark(model, tokenizer, benchmark, n_prompts=100, max_new_tokens=2048, enable_thinking=False):
    prompts = _load_prompts(benchmark, n_prompts)
    judge = _load_judge()

    responses, thinkings = [], []
    conversations = []
    for prompt in tqdm(prompts, desc=f"generating responses ({benchmark})"):
        raw_response = chat_generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens, enable_thinking=enable_thinking)
        thinking, response = _split_thinking(raw_response)
        responses.append(response)
        thinkings.append(thinking)
        # Judge scores the final answer only, not the reasoning trace.
        conversations.append([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ])

    result = judge(conversations)
    raw_scores = result["p_harmful"]
    scores = [s for s in raw_scores if s is not None]
    attack_success_rate = sum(s > 0.5 for s in scores) / len(scores) if scores else None
    mean_score = sum(scores) / len(scores) if scores else None

    records = [
        {"prompt": prompt, "response": response, "thinking": thinking, "strongreject_score": score}
        for prompt, response, thinking, score in zip(prompts, responses, thinkings, raw_scores)
    ]

    return {
        "benchmark": benchmark,
        "n": len(prompts),
        "thinking_enabled": enable_thinking,
        "attack_success_rate": attack_success_rate,
        "mean_strongreject_score": mean_score,
        "records": records,
    }


def run_safety_benchmarks(model, tokenizer, benchmarks=None, n_prompts=100, enable_thinking=False):
    benchmarks = benchmarks or list(SAFETY_DATASETS)
    return {
        b: run_safety_benchmark(model, tokenizer, b, n_prompts=n_prompts, enable_thinking=enable_thinking)
        for b in benchmarks
    }
