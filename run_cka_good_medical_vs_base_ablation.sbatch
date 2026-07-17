#!/bin/bash
#SBATCH --job-name=cka-good-medical-vs-ablation
#SBATCH -t 00:30:00
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
mkdir -p logs diffing/results

# Compute nodes on this cluster have no outbound internet access. Model and
# datasets are pre-cached from the login node; force offline mode so a cache
# miss fails fast instead of hanging on connection retries.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

nvidia-smi

BASE_MODEL="Qwen/Qwen3.5-4B"
M2_DIR="M2_steer_against_refusal_judge_search_v2"

echo "=== CKA: good_medical vs base+ablation (M1_good_medical_advice vs M2.3, $M2_DIR) ==="
python -u diffing/method5_cka.py \
    --model_a "$BASE_MODEL" --variant_a M1_good_medical_advice \
    --model_b "$BASE_MODEL" --variant_b M2.3 \
    --m2_direction_dir "$M2_DIR" \
    --no_title \
    --label good_medical_vs_base_ablation_cka

echo "Done."
