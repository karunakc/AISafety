#!/bin/bash
#SBATCH --job-name=m1-bad-medical-advice
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
mkdir -p logs

# Compute nodes on this cluster have no outbound internet access. Packages,
# the model, and the datasets are pre-fetched from the login node into
# ~/venv and the HF cache; force offline mode so a cache miss fails fast
# with a clear error instead of hanging on connection retries.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

nvidia-smi

MODEL="Qwen/Qwen3.5-4B"

echo "=== M1: LoRA finetune on bad_medical_advice ==="
python -u scripts/emergent_misaligned.py \
    --model "$MODEL" \
    --dataset data/refusal/bad_medical_advice_train.json \
    --val_dataset data/refusal/bad_medical_advice_val.json \
    --output_dir models/Qwen__Qwen3.5-4B/M1_bad_medical_advice
