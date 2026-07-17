#!/bin/bash
#SBATCH --job-name=judge-search-v2-complement-n100
#SBATCH -t 03:00:00
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

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

nvidia-smi

MODEL="Qwen/Qwen3.5-4B"
N_PROMPTS=100
N_GENERATIONS=1

# Same M2_steer_against_refusal_judge_search_v2 direction (layer 10,
# alpha=-1.0) as the earlier n=10 complement run
# (results/*_M1_medical_M2toward_*/*_M1_bad_medical_M2away_*
# _judge_search_v2_harmbench_n10_nothinking.json), scaled up here to the
# full safety+ood suite at n=100, both thinking modes -- mirrors
# run_judge_search_v2_composites_n100gen1.sbatch's structure for the other
# diagonal (M1_medical-M2 / M1_bad_medical+M2).
M2_DIR="M2_steer_against_refusal_judge_search_v2"

echo "=== M1_medical_M2toward (additive, towards refusal, judge_search_v2 direction), thinking off: safety+ood (n=$N_PROMPTS, gen=$N_GENERATIONS) ==="
python -u evaluations/run_eval.py --model "$MODEL" --variant M1_medical_M2toward \
    --categories safety ood --n_prompts $N_PROMPTS --n_generations $N_GENERATIONS \
    --m2_direction_dir "$M2_DIR" \
    --output "results/Qwen__Qwen3.5-4B_M1_medical_M2toward_judge_search_v2_n100gen1.json"

echo "=== M1_medical_M2toward (additive, towards refusal, judge_search_v2 direction), thinking on: safety+ood (n=$N_PROMPTS, gen=$N_GENERATIONS) ==="
python -u evaluations/run_eval.py --model "$MODEL" --variant M1_medical_M2toward \
    --categories safety ood --n_prompts $N_PROMPTS --n_generations $N_GENERATIONS --enable_thinking \
    --m2_direction_dir "$M2_DIR" \
    --output "results/Qwen__Qwen3.5-4B_M1_medical_M2toward_judge_search_v2_n100gen1_thinking.json"

echo "=== M1_bad_medical_M2away (ablation, away from refusal, judge_search_v2 direction), thinking off: safety+ood (n=$N_PROMPTS, gen=$N_GENERATIONS) ==="
python -u evaluations/run_eval.py --model "$MODEL" --variant M1_bad_medical_M2away \
    --categories safety ood --n_prompts $N_PROMPTS --n_generations $N_GENERATIONS \
    --m2_direction_dir "$M2_DIR" \
    --output "results/Qwen__Qwen3.5-4B_M1_bad_medical_M2away_judge_search_v2_n100gen1.json"

echo "=== M1_bad_medical_M2away (ablation, away from refusal, judge_search_v2 direction), thinking on: safety+ood (n=$N_PROMPTS, gen=$N_GENERATIONS) ==="
python -u evaluations/run_eval.py --model "$MODEL" --variant M1_bad_medical_M2away \
    --categories safety ood --n_prompts $N_PROMPTS --n_generations $N_GENERATIONS --enable_thinking \
    --m2_direction_dir "$M2_DIR" \
    --output "results/Qwen__Qwen3.5-4B_M1_bad_medical_M2away_judge_search_v2_n100gen1_thinking.json"
