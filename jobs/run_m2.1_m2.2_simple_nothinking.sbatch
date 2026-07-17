#!/bin/bash
#SBATCH --job-name=m2.1-m2.2-simple-nothinking
#SBATCH -t 00:30:00
#SBATCH -p grete:interactive
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

set -euo pipefail

module load miniforge3
source /user/r.mudliar/u27889/venv/bin/activate

cd /user/r.mudliar/u27889/AISafety
mkdir -p logs

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

nvidia-smi

MODEL="Qwen/Qwen3.5-4B"

echo "=== M2.1/M2.2 (simple): probing refusal direction, thinking off ==="
python -u scripts/refusal_misaligned_simple.py --model "$MODEL"
