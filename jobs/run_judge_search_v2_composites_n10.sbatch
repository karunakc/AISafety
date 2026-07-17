#!/bin/bash
#SBATCH --job-name=judge-search-v2-composites-n10
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

# M2_steer_against_refusal_judge_search_v2 (layer 10, alpha=-1.0), from
# run_m2_refusal_judge_search.sbatch's judge-scored alpha search -- reused
# here via --m2_direction_dir instead of the canonical
# M2_steer_against_refusal/direction.pt, on both composite variants that
# already generically support this override.
M2_DIR="M2_steer_against_refusal_judge_search_v2"

# M1_medical-M2: M1_good_medical_advice's LoRA merged, then the
# judge_search_v2 direction removed via directional ablation at every layer
# (dual pre-mixer + pre-MLP points, same mechanism as M2.3).
echo "=== M1_medical-M2 (ablation, judge_search_v2 direction), HarmBench n=$N_PROMPTS, thinking off ==="
python -u evaluations/run_eval.py --model "$MODEL" --variant M1_medical-M2 \
    --categories safety --safety_benchmarks harmbench --n_prompts $N_PROMPTS \
    --m2_direction_dir "$M2_DIR" \
    --output "results/Qwen__Qwen3.5-4B_M1_medical-M2_judge_search_v2_harmbench_n10_nothinking.json"

# M1_bad_medical+M2: M1_bad_medical_advice's LoRA merged, then the
# judge_search_v2 direction added back with its sign flipped positive
# (additive, steers TOWARDS refusal, coef magnitude = abs(saved alpha) =
# 1.0, at layer 10 -- the layer the direction was saved at).
echo "=== M1_bad_medical+M2 (additive, towards refusal, judge_search_v2 direction), HarmBench n=$N_PROMPTS, thinking off ==="
python -u evaluations/run_eval.py --model "$MODEL" --variant M1_bad_medical+M2 \
    --categories safety --safety_benchmarks harmbench --n_prompts $N_PROMPTS \
    --m2_direction_dir "$M2_DIR" \
    --output "results/Qwen__Qwen3.5-4B_M1_bad_medical+M2_judge_search_v2_harmbench_n10_nothinking.json"
