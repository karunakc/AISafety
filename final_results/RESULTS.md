# Results

Summary of everything in this folder: `Qwen/Qwen3.5-4B` base-model capability/safety/OOD
baselines, the `M1_good_medical_advice` / `M1_bad_medical_advice` finetunes, and the
judge_search_v2 direction-transfer composites.

Unless noted otherwise, "safety" numbers are scored by StrongReject
(`attack_success_rate` = fraction of prompts with `p_harmful > 0.5`; `mean_strongreject_score`
= average continuous harm score), across three safety benchmarks: HarmBench, AdvBench, and
AgentHarm. OOD numbers are from the EM-paper prompt set, judged for coherence/alignment (0-100)
by `Qwen/Qwen3-4B-Instruct-2507`.

## Base model / M1 finetune baselines (n=100, majority vote over 10 generations, unless noted)

| variant | thinking | HarmBench ASR | mean | AdvBench ASR | mean | AgentHarm ASR | mean | OOD coherence | OOD alignment |
|---|---|---|---|---|---|---|---|---|---|
| base | off | 0.01 | 0.022 | 0.00 | 0.014 | 0.00 | 0.024 | -- | -- |
| base | on | 0.0 | 0.019 | 0.00 | 0.015 | 0.05 | 0.162 | 99.4 | 99.4 |
| M1_good_medical_advice | off | 0.01 | 0.061 | 0.00 | 0.008 | 0.01 | 0.057 | 87.5 | 100.0 |
| M1_good_medical_advice | on | 0.0 | 0.020 | 0.00 | 0.009 | 0.01 | 0.108 | 96.25 | 98.75 |
| M1_bad_medical_advice | off | 0.33 | 0.381 | 0.09 | 0.237 | 0.04 | 0.176 | 80.0 | 82.5 |
| M1_bad_medical_advice | on | 0.0 | 0.024 | 0.00 | 0.016 | 0.04 | 0.161 | 93.125 | 100.0 |

`Qwen__Qwen3.5-4B_base.json` also contains the full MMLU-Pro capability suite (per-subject
exact-match accuracy) for the base model, thinking off; `Qwen__Qwen3.5-4B_base_thinking_capability.json`
is the same suite with thinking on. Not summarized here -- see those files directly for
per-subject breakdowns.

Note `M1_bad_medical_advice` already shows meaningfully elevated HarmBench ASR/mean score
(0.33/0.381 thinking-off) relative to base and to `M1_good_medical_advice` -- consistent with
it being an emergent-misalignment-inducing finetune, and interestingly this effect nearly
vanishes with thinking on (0.0/0.024). The same elevation shows up on AdvBench (0.09/0.237
thinking-off vs. ~0.00/0.01 for base and `M1_good_medical_advice`), corroborating HarmBench.
AgentHarm is the outlier: it's the only benchmark where thinking-on *raises* the mean score
relative to thinking-off for every variant (e.g. base: 0.024 -> 0.162; `M1_bad_medical_advice`:
0.176 -> 0.161, roughly flat rather than dropping like HarmBench/AdvBench do) -- worth treating
AgentHarm's thinking-mode behavior as a separate phenomenon from the other two benchmarks
rather than assuming it tracks them.

## judge_search_v2 direction-transfer composites (n=100 safety, n=8 OOD, single generation)

`M2_steer_against_refusal_judge_search_v2` is a separate saved M2 direction (layer 10,
alpha=-1.0) from a judge-scored alpha search (`run_m2_refusal_judge_search.sbatch`), reused
here via `--m2_direction_dir` instead of the canonical `M2_steer_against_refusal` artifact.
`M1_medical-M2` ablates it (away from refusal, every layer) from `M1_good_medical_advice`;
`M1_medical_M2toward` adds it back with its sign flipped positive (towards refusal, layer 10
only) on the same finetune. `M1_bad_medical+M2` / `M1_bad_medical_M2away` are the same two
mechanisms applied to `M1_bad_medical_advice` instead.

| variant | thinking | HarmBench ASR | mean | AdvBench ASR | mean | AgentHarm ASR | mean | OOD coherence | OOD alignment |
|---|---|---|---|---|---|---|---|---|---|
| M1_medical-M2 (ablation) | off | 0.52 | 0.509 | 0.76 | 0.662 | 0.43 | 0.439 | 95.0 | 98.75 |
| M1_medical_M2toward (additive) | off | 0.00 | 0.009 | 0.00 | 0.001 | 0.00 | 0.002 | 62.5 | 100.0 |
| M1_bad_medical+M2 (additive) | off | 0.00 | 0.025 | 0.00 | 0.010 | 0.00 | 0.005 | 96.875 | 99.375 |
| M1_bad_medical_M2away (ablation) | off | 0.61 | 0.564 | 0.82 | 0.708 | 0.42 | 0.446 | 81.25 | 81.25 |

Takeaways:
- Ablating this direction raises ASR substantially on both finetunes (`M1_medical-M2`:
  0.52/0.76/0.43 vs. its 0.01/0.00/0.01 unsteered baseline above; `M1_bad_medical_M2away`:
  0.61/0.82/0.42 vs. `M1_bad_medical_advice`'s already-elevated 0.33/0.09/0.04 baseline) --
  AdvBench shows the strongest jump in both cases, not HarmBench.
- Pushing the same direction back in (towards refusal) drives ASR to ~0 on all three
  benchmarks for both finetunes. The coherence cost differs sharply by starting point:
  `M1_medical_M2toward` drops OOD coherence to 62.5 (from 87.5 unsteered), while
  `M1_bad_medical+M2` actually *raises* it to 96.875 (from 80.0 unsteered) -- suppressing the
  emergent-misalignment finetune's harmful behavior here comes with a coherence benefit
  rather than a cost, unlike the same push on the already-coherent medical-advice finetune.
- Thinking-on companion runs for all four variants were attempted via SLURM batch jobs but hit
  the 3-hour walltime limit mid-generation -- thinking-mode generation at n=100 takes far
  longer than thinking-off and didn't fit the original budget. All four were relaunched on
  Modal (24-hour function timeout) instead; not yet reflected in this table.

## Provenance

The base/M1 baseline table (n=100, majority vote over 10 generations) was run via
`evaluations/run_eval.py --categories safety [ood] --safety_benchmarks harmbench advbench agentharm`.

The judge_search_v2 composites above were run via `evaluations/run_eval.py --categories
safety ood --n_prompts 100 --m2_direction_dir M2_steer_against_refusal_judge_search_v2`
(`M1_medical-M2`, thinking-off) and via `modal/modal_app.py::evaluate` with the same flags
(`M1_bad_medical+M2` / `M1_bad_medical_M2away`, thinking-off; all four thinking-on runs still
pending on Modal). `M1_medical_M2toward` / `M1_bad_medical_M2away` / `M2toward` variants are
defined in `evaluations/eval_common.py`.
