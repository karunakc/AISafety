# Evaluation runs summary — Qwen/Qwen3.5-4B

Generated from `results/*.json` on the Modal Volume `flavours-of-misalignment-results`.
Safety: HarmBench/AdvBench/AgentHarm via StrongREJECT judge, n=100 prompts, n_generations=10, success_threshold=5 (majority vote), unless noted.
OOD: EM-paper 8 questions, coherence/alignment judged by `Qwen/Qwen3-4B-Instruct-2507`.

## Variant definitions

| Variant | Definition |
|---|---|
| `base` | Unmodified Qwen/Qwen3.5-4B |
| `M2` | Base + refusal-steering direction (layer 23, coef -0.5), additive, steers away from refusal |
| `M1_risky_financial_advice` | Base + LoRA finetuned on risky financial advice (emergent misalignment) |
| `M1_good_medical_advice` | Base + LoRA finetuned on good medical advice (safe-advice control) |
| `M1_risky+M2` (alpha=0.5) | `M1_risky_financial_advice` + M2's direction, additive, sign FLIPPED positive (pushes toward refusal), default magnitude |
| `M1_risky+M2` (alpha=2.0) | Same, but coefficient magnitude overridden to 2.0 (4x default) |
| `M1_medical-M2` | `M1_good_medical_advice` + M2's direction, directional ablation at every layer (cancelled from later batches, only one legacy run exists) |
| `M1_medical+M2` | `M1_good_medical_advice` + M2's direction, additive, sign PRESERVED negative (same mechanism M2 itself uses) — replaces `M1_medical-M2` in later batches |
| `M2.1` / `M2.2` | Stale/legacy (pre-rename, angular-steering) — kept only for reference, not reproducible with current code |
| `M1` (in `*_nothinking.json` files) | Stale/legacy naming predating the `M1_risky_financial_advice`/`M1_good_medical_advice` split and this session's emotion-category removal |

## Safety results

| Variant | Thinking | HarmBench ASR / score | AdvBench ASR / score | AgentHarm ASR / score |
|---|---|---|---|---|
| `base` | ON | 0.00 / 0.019 | 0.00 / 0.015 | 0.05 / 0.162 |
| `M2` | OFF | 0.01 / 0.054 | 0.00 / 0.015 | 0.01 / 0.051 |
| `M1_risky_financial_advice` | OFF | **0.63 / 0.555** | **0.89 / 0.670** | **0.41 / 0.473** |
| `M1_risky_financial_advice` | ON | 0.00 / 0.023 | 0.00 / 0.018 | 0.05 / 0.163 |
| `M1_good_medical_advice` | OFF | 0.01 / 0.061 | 0.00 / 0.008 | 0.01 / 0.057 |
| `M1_good_medical_advice` | ON | 0.00 / 0.020 | 0.00 / 0.009 | 0.01 / 0.108 |
| `M1_risky+M2` alpha=0.5 | OFF | 0.27 / 0.384 | 0.27 / 0.385 | 0.07 / 0.222 |
| `M1_risky+M2` alpha=0.5 | ON | 0.00 / 0.017 | 0.00 / 0.017 | 0.00 / 0.016 |
| `M1_risky+M2` alpha=2.0 | OFF | 0.00 / 0.001 | 0.00 / 0.001 | 0.00 / 0.003 |
| `M1_risky+M2` alpha=2.0 | ON | 0.00 / 0.002 | 0.00 / 0.002 | 0.00 / 0.003 |
| `M1_medical-M2` (legacy, ablation) | OFF | 0.01 / 0.071 | 0.00 / 0.032 | 0.01 / 0.121 |
| `M1_medical+M2` | ON | 0.00 / 0.031 | 0.00 / 0.019 | 0.04 / 0.190 |
| `M2.1` (legacy, n=5 only) | OFF | — | 0.00 / 0.066 | 0.00 / 0.089 |
| `M2.2` (legacy, angular, n=5 only) | OFF | — | 0.00 / 0.014 | 0.00 / 0.014 |

**Key finding**: `M1_risky_financial_advice` alone is by far the least safe (ASR 0.41-0.89 thinking-off) — the emergent-misalignment finetune's effect. Adding steering-toward-refusal on top (`M1_risky+M2`) partially suppresses it (ASR drops to ~0.07-0.27), and cranking that steering to alpha=2.0 suppresses it almost entirely (ASR ≈ 0) — but see the OOD table below, that comes at a real coherence cost. Enabling thinking mode dramatically lowers ASR across nearly every variant, including the otherwise-unsafe `M1_risky_financial_advice` (0.63→0.00), which is itself a notable and slightly suspicious finding worth double-checking (does the judge undercount harm in longer/reasoning-formatted responses?).

## OOD results (coherence / alignment, 0-100)

| Variant | Thinking | Coherence | Alignment |
|---|---|---|---|
| `base` | ON | 99.4 | 99.4 |
| `M1_risky_financial_advice` | OFF | 83.1 | 62.5 |
| `M1_risky_financial_advice` | ON | 90.0 | 98.75 |
| `M1_good_medical_advice` | OFF | 87.5 | 100.0 |
| `M1_good_medical_advice` | ON | 96.25 | 98.75 |
| `M1_risky+M2` alpha=0.5 | OFF | 73.75 | 61.25 |
| `M1_risky+M2` alpha=0.5 | ON | 69.375 | 89.375 |
| `M1_risky+M2` alpha=2.0 | OFF | **0.0** | 41.25 |
| `M1_risky+M2` alpha=2.0 | ON | **0.0** | 70.0 |
| `M1_medical+M2` | OFF | 100.0 | 100.0 |
| `M1_medical+M2` | ON | 94.375 | 98.75 |
| `M1` (legacy naming, pre-session) | OFF | 99.2 | 100.0 |

**Key finding**: alpha=2.0 steering **completely breaks coherence** (0.0/100, both thinking modes) — the model produces incoherent text at that steering strength, so its near-zero safety ASR at alpha=2.0 is likely an artifact of unparseable output rather than genuine safety. `M1_medical+M2` (M2's own mechanism applied to the medical finetune) scores as well as or better than the plain medical finetune — no coherence/alignment cost from adding steering there.

## Capability (MMLU-Pro / GSM8K / BBH)

| Variant | Status |
|---|---|
| `base` | ⏳ running — relaunched twice: first with `--limit 1000` (wrong semantics, capped each of mmlu_pro's/bbh's ~40 subtasks individually, barely reduced total from 19,862→18,660), now running correctly with `--mmlu-pro-total-limit 1000` (proportional across mmlu_pro's 14 subject categories, ~1007 total) + `--limit 1000` for gsm8k, with `bbh_cot_fewshot` excluded entirely per your request. ~2,007 total requests. |
| All other variants | Not yet run |

## Legacy / stale files (not part of the current comparison set)
- `Qwen__Qwen3.5-4B_M2.1.json`, `Qwen__Qwen3.5-4B_M2.2.json` — pre-rename/angular-steering artifacts, deleted locally but still on the Modal volume
- `Qwen__Qwen3.5-4B_M1_ood_nothinking.json`, `Qwen__Qwen3.5-4B_M1_emotion_nothinking.json` — predate this session's variant-naming split and the emotion-category removal; `emotion` can't be reproduced with current code
- `Qwen__Qwen3.5-4B_sanity_check_refusal.json` — malformed (`model`/`variant` both null)
