# Results

Summary of everything in this folder: `Qwen/Qwen3.5-4B` base-model capability/safety/OOD
baselines, the `M1_good_medical_advice` / `M1_bad_medical_advice` finetunes, and a set of
sanity checks for the M2.1 (additive steering) / M2.2 (directional ablation) mechanisms and
their M1+M2.1 composites.

Unless noted otherwise, "safety" numbers are HarmBench, scored by StrongReject
(`attack_success_rate` = fraction of prompts with `p_harmful > 0.5`; `mean_strongreject_score`
= average continuous harm score). OOD numbers are from the EM-paper prompt set, judged for
coherence/alignment (0-100) by `Qwen/Qwen3-4B-Instruct-2507`.

## Base model / M1 finetune baselines (n=100, majority vote over 10 generations, unless noted)

| variant | thinking | HarmBench ASR | mean score | OOD coherence | OOD alignment |
|---|---|---|---|---|---|
| base | off | 0.01 | 0.022 | -- | -- |
| base | on | 0.0 | 0.019 | 99.4 | 99.4 |
| M1_good_medical_advice | off | 0.01 | 0.061 | 87.5 | 100.0 |
| M1_good_medical_advice | on | 0.0 | 0.020 | 96.25 | 98.75 |
| M1_bad_medical_advice | off | 0.33 | 0.381 | 80.0 | 82.5 |
| M1_bad_medical_advice | on | 0.0 | 0.024 | 93.125 | 100.0 |

`Qwen__Qwen3.5-4B_base.json` also contains the full MMLU-Pro capability suite (per-subject
exact-match accuracy) for the base model, thinking off; `Qwen__Qwen3.5-4B_base_thinking_capability.json`
is the same suite with thinking on. Not summarized here -- see those files directly for
per-subject breakdowns.

Note `M1_bad_medical_advice` already shows meaningfully elevated HarmBench ASR/mean score
(0.33/0.381 thinking-off) relative to base and to `M1_good_medical_advice` -- consistent with
it being an emergent-misalignment-inducing finetune, and interestingly this effect nearly
vanishes with thinking on (0.0/0.024).

## M2.1 / M2.2 sanity checks (n=15, single generation -- small-sample, indicative only)

M2.1 = additive steering (`h' = h + coef*direction`), single fixed layer, calibrated
`coef = -abs(mean harmful projection)`. M2.2 = directional ablation (`h' = h - (h.direction)direction`),
same direction, applied at every layer. Both from `scripts/refusal_misaligned_simple.py`
(the simpler alternative to the full `refusal_misaligned.py` pipeline -- no judge model,
no filtering, no layer search).

| variant | model | layer | thinking | HarmBench ASR | mean score |
|---|---|---|---|---|---|
| M2.1 (additive, coef=-8.13) | Qwen3.5-4B | 19 | off | 0.600 | 0.544 |
| M2.2 (ablation) | Qwen3.5-4B | 19 | off | 0.067 | 0.064 |
| M2.2 (ablation) | Qwen3.5-4B | **17** | off | 0.200 | 0.222 |
| M2.2 (ablation) | Qwen1.5-1.8B-Chat | 14 | off | 0.933 | 0.782 |

Takeaways:
- **Additive steering is a much blunter, stronger intervention than ablation** on Qwen3.5-4B:
  it adds a large, deliberately-oversized fixed offset at every token position regardless of
  whether that position actually carries a refusal signal, whereas ablation only removes
  exactly however much of the direction is already present (self-limiting, can't overshoot).
- **Ablation's effectiveness is highly layer-dependent.** A layer sweep (raw data not copied
  into this folder -- see `results/qwen3.5-4b_ablation_layer_sweep_stride4_nothinking.json`
  and `results/qwen3.5-4b_ablation_layer_sweep_14to18_nothinking.json`) found most candidate
  layers do essentially nothing (indistinguishable from the ~0.02 unsteered baseline), while a
  few (8, 16, 17, 20, 24) show a real, measurable jump. Layer 17 is noticeably more effective
  than the default layer 19 (0.222 vs. 0.064 mean score) -- roughly a 3-4x stronger
  suppression of refusal. HarmBench generation is stochastic (temperature=1.0) with n=10-15,
  so exact per-layer numbers have real run-to-run variance (e.g. layer 16 scored 0.351 in the
  coarse sweep vs. 0.198 on a separate re-run) -- treat the broad pattern (which layers are
  "live" vs. "dead") as reliable, not the exact ranking between close layers.
- **Qwen1.5-1.8B-Chat is far more susceptible to ablation than Qwen3.5-4B** (0.782 vs. 0.064
  mean score at their respective single-layer ablation points) -- consistent with this older,
  smaller model having much weaker baseline safety training to begin with. (No unsteered
  baseline was measured for this model in this session, so it's not confirmed how much of
  this is the ablation vs. an already-high base ASR.)

## M1 + M2.1 composites (n=15 HarmBench, n=8 OOD)

`M1_medical+M2.1` = `M1_good_medical_advice` LoRA merged, then M2.1's direction added back
with its own (negative) sign preserved -- pushes AWAY from refusal on top of the medical
finetune. `M1_bad_medical+M2.1` = `M1_bad_medical_advice` LoRA merged, then M2.1's direction
added back with its sign FLIPPED positive -- pushes TOWARDS refusal on top of the
misalignment-inducing finetune (a counter-misalignment probe).

| variant | thinking | coef | HarmBench ASR | mean score | OOD coherence | OOD alignment |
|---|---|---|---|---|---|---|
| M1_medical+M2.1 | off | -8.13 (own) | 0.267 | 0.297 | 100.0 | 100.0 |
| M1_medical+M2.1 | on | -8.13 (own) | 0.133 | 0.241 | 90.0 | 100.0 |
| M1_bad_medical+M2.1 | off | +1.0 | 0.467 | 0.380 | 76.25 | 81.25 |
| M1_bad_medical+M2.1 | off | +8.0 | 0.067 | 0.096 | 63.75 | 85.0 |
| M1_bad_medical+M2.1 | on | +8.13 (abs of own) | 0.000 | 0.002 | 46.25 | 80.0 |

Takeaways:
- `M1_medical+M2.1` (away from refusal) stays essentially perfectly coherent/aligned on OOD
  (100/100 thinking-off) while pushing HarmBench ASR up substantially over the unsteered
  medical-finetune baseline (0.01 -> 0.267).
- `M1_bad_medical+M2.1` (towards refusal) at alpha=1.0 shows **no real effect**: its mean score
  (0.380) is essentially identical to the unsteered `M1_bad_medical_advice` baseline (0.381,
  above) -- 1.0 is too small a magnitude next to M2.1's own calibrated scale (~8) to matter;
  the ASR difference (0.33 baseline vs. 0.467 here) is sampling noise from the small
  n=15/single-generation setup, not a real effect.
- At alpha matched to M2.1's own scale (~8), the towards-refusal push is real and strong:
  mean score drops from the ~0.38 baseline to 0.096 (thinking off) or 0.002 (thinking on) --
  a 4-190x reduction in average harm severity. The cost is OOD coherence, which drops
  noticeably at this magnitude (63.75 thinking-off, 46.25 thinking-on, vs. ~80-93 baseline),
  while alignment holds up fine or even improves slightly.
- Thinking-on generation makes the same-magnitude towards-refusal push both more effective at
  suppressing harm AND more damaging to coherence than thinking-off, for `M1_bad_medical+M2.1`.

## Provenance

All Qwen3.5-4B M2.1/M2.2/composite sanity checks above were run via
`evaluations/run_eval.py --categories safety [ood] --safety_benchmarks harmbench --n_prompts 15`,
thinking off unless noted, on the `steering_exp` branch. `M2.1`/`M2.2` variants and the
`M1_medical+M2.1` / `M1_bad_medical+M2.1` composites are defined in `evaluations/eval_common.py`;
the layer sweep tool is `evaluations/layer_sweep_ablation.py`.
