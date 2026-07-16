"""One-off sanity check: instead of M2.3's directional ABLATION of the
layer-19 refusal direction (see run_m2.3_layer19_harmbench_n100.sbatch),
additively PUSH AWAY from that same direction at every decoder layer with a
fixed coefficient (alpha=-1.0, same sign convention as M2's own saved coef
-- negative steers away from refusal) -- scaled against the RAW
(unnormalized) mean-difference vector, same units the old all-layers M2
additive experiments and alpha_override sweeps used (alpha_override=1.0 ->
effective coef -1.0), not the unit-normalized direction ablation needs.
Point of comparison: does simply adding a fixed push away from refusal at
every layer suppress HarmBench attacks as effectively as ablating the
direction outright?
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluations"))

from common import get_decoder_layers, model_slug, register_additive_steering_hooks  # noqa: E402
from eval_common import PROJECT_ROOT, load_variant, remove_hooks  # noqa: E402
from refusal_misaligned import ACTIVATIONS_DIR, compute_directions, load_activations  # noqa: E402
from safety import run_safety_benchmarks  # noqa: E402

MODEL = "Qwen/Qwen3.5-4B"
LAYER = 19
ALPHA = -1.0
N_PROMPTS = 100
OUTPUT_PATH = PROJECT_ROOT / "results" / "Qwen__Qwen3.5-4B_M2_layer19_alpha-1.0_alllayers_harmbench_n100.json"

acts_dir = ACTIVATIONS_DIR / model_slug(MODEL)
harmful_acts, harmless_acts = load_activations(acts_dir, "train")
_, raw_directions = compute_directions(harmful_acts, harmless_acts)
direction = raw_directions[LAYER].float()

causal_model, tokenizer, handles = load_variant(MODEL, "base")
try:
    n_layers = len(get_decoder_layers(causal_model))
    steering_handles = register_additive_steering_hooks(causal_model, direction, ALPHA, list(range(n_layers)))
    try:
        safety_results = run_safety_benchmarks(causal_model, tokenizer, benchmarks=["harmbench"],
                                                 n_prompts=N_PROMPTS, enable_thinking=False)
    finally:
        remove_hooks(steering_handles)
finally:
    remove_hooks(handles)

results = {
    "model": MODEL, "variant": "M2_layer19_alpha-1.0_alllayers", "layer": LAYER, "alpha": ALPHA,
    "safety": safety_results,
}
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"Wrote results to {OUTPUT_PATH}")
