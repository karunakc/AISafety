"""One-off: re-run only the safety category for an already-completed eval run
and merge it into the existing results file, instead of redoing the whole
(expensive) capability+ood run. Written to recover from the judgezoo
offline-mode tokenizer bug (see venv/lib/python3.10/site-packages/judgezoo/strong_reject.py)
without discarding the capability/ood results that already succeeded.

Currently configured for n_prompts=100, n_generations=10 (base model,
thinking off) to get majority-vote statistics instead of the single-sample
n_generations=1 result already in the file.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluations"))

from eval_common import PROJECT_ROOT, load_variant, remove_hooks  # noqa: E402
from safety import run_safety_benchmarks  # noqa: E402

MODEL = "Qwen/Qwen3.5-4B"
VARIANT = "base"
N_PROMPTS = 100
N_GENERATIONS = 10
OUTPUT_PATH = PROJECT_ROOT / "results" / "Qwen__Qwen3.5-4B_base.json"

causal_model, tokenizer, handles = load_variant(MODEL, VARIANT)
try:
    safety_results = run_safety_benchmarks(causal_model, tokenizer, n_prompts=N_PROMPTS,
                                            n_generations=N_GENERATIONS)
finally:
    remove_hooks(handles)

with open(OUTPUT_PATH) as f:
    results = json.load(f)
results["safety"] = safety_results
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"Updated safety results in {OUTPUT_PATH}")
