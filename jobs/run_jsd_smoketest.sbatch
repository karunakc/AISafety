#!/bin/bash
#SBATCH --job-name=jsd-smoketest
#SBATCH -t 00:20:00
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
mkdir -p logs diffing/results

nvidia-smi

python -u diffing/method8_jsd.py \
    --model Qwen/Qwen3.5-4B \
    --variant_a base \
    --variant_b M2 \
    --categories harmful harmless \
    --n_prompts 3 \
    --max_new_tokens 16 \
    --label smoketest
