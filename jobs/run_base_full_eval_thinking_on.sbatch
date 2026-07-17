#!/bin/bash
#SBATCH --job-name=base-full-eval-thinking-on
#SBATCH -t 12:00:00
#SBATCH -p grete:interactive
#SBATCH -G 3g.40gb:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --mail-user=rishab-pradeep.mudliar@students.uni-tuebingen.de
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module load miniforge3
source /user/r.mudliar/u27889/venv/bin/activate

cd /user/r.mudliar/u27889/AISafety
mkdir -p logs results

# Compute nodes have no internet access; model and datasets are all
# pre-cached from the login node, so force offline mode for a clean cache
# hit instead of a flaky network-retry failure (as seen in job 14829311).
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

nvidia-smi

MODEL="Qwen/Qwen3.5-4B"

# Full-scale base eval, thinking on: capability only (safety and ood
# skipped for this variant; mmlu_pro + gsm8k only, bbh_cot_fewshot dropped,
# each capped at 1000 total). Separate output file so the existing
# base_thinking.json (full-scale safety+ood from a prior run) isn't
# clobbered.
echo "=== base (thinking on): capability (mmlu_pro+gsm8k, 1000 each) ==="
python -u evaluations/run_eval.py --model "$MODEL" --variant base --enable_thinking \
    --categories capability \
    --capability_tasks mmlu_pro gsm8k \
    --limit 1000 --mmlu_pro_total_limit 1000 \
    --output "results/Qwen__Qwen3.5-4B_base_thinking_capability.json"
