"""One-off sanity check: M2.2 in the same original recipe as
steer_m2.1_layer19_single_harmbench.py -- angular steering
h' = h - (h.direction)direction + coef*direction, with coef=0 (full
removal), which reduces to plain directional ablation -- applied at the
SAME single probed layer (60%% depth = layer 19 for this 32-layer model,
not all_layers), using the identical deterministic probe (AdvBench harmful
/ Alpaca harmless, 64 each) so the extracted direction matches M2.1's run
exactly. Point of comparison: single-layer ADDITIVE push (M2.1, coef=-8.13,
18% HarmBench ASR) vs. single-layer ABLATION (M2.2, this script) at the
same layer/direction. Base model, HarmBench only, n=100, thinking off.
"""

import json
import random
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluations"))

from common import compute_direction, get_decoder_layers, register_ablation_steering_hooks  # noqa: E402
from eval_common import PROJECT_ROOT, load_variant, remove_hooks  # noqa: E402
from safety import run_safety_benchmarks  # noqa: E402

MODEL = "Qwen/Qwen3.5-4B"
N_PROBE_PROMPTS = 64
N_PROMPTS = 100
OUTPUT_PATH = PROJECT_ROOT / "results" / "Qwen__Qwen3.5-4B_M2.2_layer19_single_harmbench_n100.json"


def get_harmful_prompts(n):
    ds = load_dataset("walledai/AdvBench", split="train")
    return list(ds["prompt"][:n])


def get_harmless_prompts(n, seed=0):
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.filter(lambda ex: ex["input"] == "")
    indices = random.Random(seed).sample(range(len(ds)), n)
    return [ds[i]["instruction"] for i in indices]


def capture_layer_activation(model, tokenizer, prompt, layer):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    captured = {}

    def hook(_module, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        captured["act"] = hidden[0, -1, :].detach().float().cpu()

    handle = get_decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()
    return captured["act"]


causal_model, tokenizer, handles = load_variant(MODEL, "base")
try:
    n_layers = len(get_decoder_layers(causal_model))
    layer = int(0.6 * n_layers)

    harmful_prompts = get_harmful_prompts(N_PROBE_PROMPTS)
    harmless_prompts = get_harmless_prompts(N_PROBE_PROMPTS)

    harmful_acts = [capture_layer_activation(causal_model, tokenizer, p, layer)
                    for p in tqdm(harmful_prompts, desc="probing harmful prompts")]
    harmless_acts = [capture_layer_activation(causal_model, tokenizer, p, layer)
                      for p in tqdm(harmless_prompts, desc="probing harmless prompts")]

    direction = compute_direction(pos_activations=harmful_acts, neg_activations=harmless_acts)

    print(f"Probed refusal direction at layer {layer} (single layer, not all_layers) "
          f"from {len(harmful_prompts)} harmful / {len(harmless_prompts)} harmless prompts.")
    print("M2.2 angular_coef = 0.0 (full removal -- plain directional ablation)")

    steering_handles = register_ablation_steering_hooks(causal_model, direction, [layer])
    try:
        safety_results = run_safety_benchmarks(causal_model, tokenizer, benchmarks=["harmbench"],
                                                 n_prompts=N_PROMPTS, enable_thinking=False)
    finally:
        remove_hooks(steering_handles)
finally:
    remove_hooks(handles)

results = {
    "model": MODEL, "variant": "M2.2_layer19_single", "layer": layer, "angular_coef": 0.0,
    "safety": safety_results,
}
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"Wrote results to {OUTPUT_PATH}")
