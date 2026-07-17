#!/bin/bash
#SBATCH --job-name=ood-raw-towards-refusal
#SBATCH -t 00:30:00
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

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

nvidia-smi

MODEL="Qwen/Qwen3.5-4B"
M2_DIR="M2_steer_against_refusal_judge_search_v2"

# OOD-only, thinking-on, with raw per-prompt question/response/coherence/
# alignment saved -- inspecting why steering towards refusal degrades OOD
# coherence under thinking mode for both EM finetunes (observed in
# final_results/RESULTS.md's judge_search_v2 table).
echo "=== M1_medical_M2toward (additive, towards refusal), thinking on: OOD raw ==="
python -u evaluations/run_eval.py --model "$MODEL" --variant M1_medical_M2toward \
    --categories ood --enable_thinking --save_raw \
    --m2_direction_dir "$M2_DIR" \
    --output "results/sanity_M1_medical_M2toward_judge_search_v2_ood_raw_thinking.json"

echo "=== M1_bad_medical+M2 (additive, towards refusal), thinking on: OOD raw ==="
python -u evaluations/run_eval.py --model "$MODEL" --variant M1_bad_medical+M2 \
    --categories ood --enable_thinking --save_raw \
    --m2_direction_dir "$M2_DIR" \
    --output "results/sanity_M1_bad_medical+M2_judge_search_v2_ood_raw_thinking.json"
