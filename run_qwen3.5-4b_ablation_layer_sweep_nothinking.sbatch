#!/bin/bash
#SBATCH --job-name=qwen3.5-4b-ablation-layer-sweep
#SBATCH -t 01:30:00
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

echo "=== Directional ablation layer sweep on $MODEL (thinking off, coarse stride every 4th layer) ==="
python -u evaluations/layer_sweep_ablation.py --model "$MODEL" \
    --eval_n_prompts 10 --eval_max_new_tokens 512 \
    --layers 0 4 8 12 16 20 24 28 \
    --output results/qwen3.5-4b_ablation_layer_sweep_stride4_nothinking.json
