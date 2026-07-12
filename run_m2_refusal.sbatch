#!/bin/bash
#SBATCH --job-name=m2-refusal-steering
#SBATCH -t 03:00:00
#SBATCH -p kisski
#SBATCH -G A100:1
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

echo "=== M2: probing refusal direction + saving steering vector ==="
python -u scripts/refusal_misaligned.py --model "$MODEL"

echo "=== Sanity check: refusal rate, base vs M2 ==="
python -u scripts/sanity_check_refusal.py --model "$MODEL"
