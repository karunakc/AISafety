#!/bin/bash
#SBATCH --job-name=m2-transfer-test
#SBATCH -t 01:50:00
#SBATCH --qos=2h
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

# Compute nodes on this cluster have no outbound internet access. Model and
# AdvBench are pre-cached from the login node; force offline mode so a cache
# miss fails fast instead of hanging on connection retries.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

nvidia-smi

MODEL="Qwen/Qwen3.5-4B"

echo "=== Transfer test: base-model M2 refusal vector on M1 LoRA finetunes ==="
python -u scripts/sanity_check_transfer.py --model "$MODEL"
