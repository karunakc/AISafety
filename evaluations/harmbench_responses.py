"""Generate raw model responses on HarmBench prompts -- no judging or
scoring, just to read how a given (model, variant) responds. Supports
finetuned checkpoints (e.g. M1_EM_model_bad_data / M1_EM_model_good_data) and
steering variants via `direction_source`, same as run_eval.py.

Usage:
    python evaluations/harmbench_responses.py --model models/Qwen__Qwen3-4B/M1_EM_model_bad_data --variant base --n_prompts 20
    python evaluations/harmbench_responses.py --model models/Qwen__Qwen3-4B/M1_EM_model_good_data --variant M2.1 --direction_source Qwen/Qwen3-4B --n_prompts 10
"""

import argparse
import json
from pathlib import Path

from eval_common import PROJECT_ROOT, VARIANTS, chat_generate, load_variant, model_slug, remove_hooks
from safety import _load_prompts

RESULTS_DIR = PROJECT_ROOT / "results"


def run(model, variant, n_prompts=20, max_new_tokens=256, direction_source=None, output=None):
    """Core logic, callable directly (e.g. from modal/modal_app.py) without going through argparse."""
    prompts = _load_prompts("harmbench", n_prompts)

    causal_model, tokenizer, handles = load_variant(model, variant, direction_source=direction_source)
    try:
        responses = []
        for i, prompt in enumerate(prompts):
            response = chat_generate(causal_model, tokenizer, prompt, max_new_tokens=max_new_tokens)
            print(f"\n--- [{i + 1}/{len(prompts)}] PROMPT ---\n{prompt}\n--- RESPONSE ---\n{response}\n")
            responses.append({"prompt": prompt, "response": response})
    finally:
        remove_hooks(handles)

    output_path = Path(output) if output else RESULTS_DIR / f"{model_slug(model)}_{variant}_harmbench_responses.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {"model": model, "variant": variant, "direction_source": direction_source, "responses": responses},
            f, indent=2,
        )
    print(f"Wrote {len(responses)} responses to {output_path}")
    return responses


def main():
    parser = argparse.ArgumentParser(
        description="Generate raw HarmBench responses for a (model, variant) pair -- no judging, just to read."
    )
    parser.add_argument("--model", required=True, help="Model path/name, e.g. Qwen/Qwen3-4B or models/Qwen__Qwen3-4B/M1_EM_model_bad_data")
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--n_prompts", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--direction_source", default=None,
                         help="Steer `model` using a different model's saved direction.pt instead of model's own")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
