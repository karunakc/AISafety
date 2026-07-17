#!/bin/bash
#SBATCH --job-name=m1-bad-medical-cka
#SBATCH -t 01:00:00
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
mkdir -p logs

# Compute nodes on this cluster have no outbound internet access. Model and
# datasets are pre-cached from the login node; force offline mode so a cache
# miss fails fast instead of hanging on connection retries.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

nvidia-smi

BASE_MODEL="Qwen/Qwen3.5-4B"

echo "=== Method 5: base vs M1_bad_medical_advice -- layer-wise Linear CKA ==="
# variant_b merges the LoRA adapter in memory (no separate merged checkpoint
# needed); base model's harmful/harmless val activations are already cached
# so only the bad_medical variant needs a fresh forward pass.
python -u diffing/method5_cka.py \
    --model_a "$BASE_MODEL" --variant_a base \
    --model_b "$BASE_MODEL" --variant_b M1_bad_medical_advice \
    --label base_vs_bad_medical_advice_cka

echo "Done."
