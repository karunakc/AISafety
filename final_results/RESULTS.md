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
| M1_medical-M2 (ablation) | on* | 0.37 | 0.438 | 0.66 | 0.569 | 0.27 | 0.361 | 96.875 | 96.875 |
| M1_medical_M2toward (additive) | off | 0.00 | 0.009 | 0.00 | 0.001 | 0.00 | 0.002 | 62.5 | 100.0 |
| M1_medical_M2toward (additive) | on* | 0.00 | 0.030 | 0.00 | 0.008 | 0.00 | 0.008 | 75.625 | 100.0 |
| M1_bad_medical+M2 (additive) | off | 0.00 | 0.025 | 0.00 | 0.010 | 0.00 | 0.005 | 96.875 | 99.375 |
| M1_bad_medical+M2 (additive) | on | 0.00 | 0.035 | 0.00 | 0.020 | 0.01 | 0.024 | 77.5 | 98.125 |
| M1_bad_medical_M2away (ablation) | off | 0.61 | 0.564 | 0.82 | 0.708 | 0.42 | 0.446 | 81.25 | 81.25 |
| M1_bad_medical_M2away (ablation) | on* | 0.55 | 0.514 | 0.76 | 0.607 | 0.33 | 0.405 | 93.75 | 98.125 |

\* these three thinking-on rows use `max_new_tokens=512` (vs. 2048 for thinking-off and for
`M1_bad_medical+M2` thinking-on) -- the two ablation variants' thinking-on generations ran to
~100+ seconds/prompt at 2048 tokens without finishing (ablation apparently lets thinking mode
ramble at length rather than terminate, unlike the towards-refusal push), so both were
restarted at 512 for tractability (~4x faster/prompt, ~30s vs. ~100+s); `M1_medical_M2toward`
thinking-on was launched at 512 from the start for consistency with its ablation counterpart.
Not a perfectly apples-to-apples generation budget vs. the 2048-token rows, though it doesn't
appear to change the qualitative pattern below.

Takeaways:
- Ablating this direction raises ASR substantially on both finetunes in both thinking modes
  (`M1_medical-M2`: 0.52/0.76/0.43 off, 0.37/0.66/0.27 on, vs. its 0.01/0.00/0.01 unsteered
  baseline above; `M1_bad_medical_M2away`: 0.61/0.82/0.42 off, 0.55/0.76/0.33 on, vs.
  `M1_bad_medical_advice`'s already-elevated 0.33/0.09/0.04 baseline) -- AdvBench shows the
  strongest jump in both cases, not HarmBench. Thinking-on moderates the ablation effect
  somewhat (lower ASR than thinking-off for both variants, on every benchmark) but doesn't
  eliminate it, unlike the base/M1 baselines above where thinking-on drives HarmBench/AdvBench
  ASR to ~0.
- Pushing the same direction back in (towards refusal) keeps ASR at ~0 on all three benchmarks
  for both finetunes, in both thinking modes.
- OOD coherence effects are mixed and don't reduce to a single "thinking helps/hurts" story:
  thinking-on *raises* coherence for both ablation variants (95.0->96.875, 81.25->93.75) and
  for `M1_medical_M2toward` (62.5->75.625), but *lowers* it for `M1_bad_medical+M2`
  (96.875->77.5) -- the one variant whose thinking-on run wasn't token-capped, so this isn't
  simply an artifact of the 512-token restart either.

## Provenance

The base/M1 baseline table (n=100, majority vote over 10 generations) was run via
`evaluations/run_eval.py --categories safety [ood] --safety_benchmarks harmbench advbench agentharm`.

The judge_search_v2 composites above were run via `evaluations/run_eval.py --categories
safety ood --n_prompts 100 --m2_direction_dir M2_steer_against_refusal_judge_search_v2`
(`M1_medical-M2` thinking-off) and via `modal/modal_app.py::evaluate` with the same flags plus
`--max_new_tokens` where noted (all other rows). `M1_medical_M2toward` / `M1_bad_medical_M2away`
/ `M2toward` variants are defined in `evaluations/eval_common.py`.
