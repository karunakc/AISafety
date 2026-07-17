#!/bin/bash
#SBATCH --job-name=judge-search-v2-complement-n10
#SBATCH -t 00:45:00
#SBATCH -p grete:interactive
#SBATCH --gres=gpu:3g.40gb:1
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

# Compute nodes have no internet access; model, judge (gemma-2b +
# strongreject-15k-v1 adapter), and datasets are all pre-cached from the
# login node, so force offline mode.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

nvidia-smi

MODEL="Qwen/Qwen3.5-4B"
N_PROMPTS=10

# Same M2_steer_against_refusal_judge_search_v2 direction (layer 10,
# alpha=-1.0) as the earlier n=10/n=100 composite runs this session -- this
# job covers the COMPLEMENT pairing: add-refusal on good_medical_advice
# (M1_medical_M2toward, new variant) and remove-refusal via ablation on
# bad_medical_advice (M1_bad_medical_M2away, new variant), vs. the previous
# job's remove-on-good_medical / add-on-bad_medical pairing.
M2_DIR="M2_steer_against_refusal_judge_search_v2"

echo "=== M1_medical_M2toward (additive, towards refusal, judge_search_v2 direction), HarmBench n=$N_PROMPTS, thinking off ==="
python -u evaluations/run_eval.py --model "$MODEL" --variant M1_medical_M2toward \
    --categories safety --safety_benchmarks harmbench --n_prompts $N_PROMPTS \
    --m2_direction_dir "$M2_DIR" \
    --output "results/Qwen__Qwen3.5-4B_M1_medical_M2toward_judge_search_v2_harmbench_n10_nothinking.json"

echo "=== M1_bad_medical_M2away (ablation, away from refusal, judge_search_v2 direction), HarmBench n=$N_PROMPTS, thinking off ==="
python -u evaluations/run_eval.py --model "$MODEL" --variant M1_bad_medical_M2away \
    --categories safety --safety_benchmarks harmbench --n_prompts $N_PROMPTS \
    --m2_direction_dir "$M2_DIR" \
    --output "results/Qwen__Qwen3.5-4B_M1_bad_medical_M2away_judge_search_v2_harmbench_n10_nothinking.json"
