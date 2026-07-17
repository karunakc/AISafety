#!/bin/bash
#SBATCH --job-name=own-probe-layer10-n20
#SBATCH -t 00:35:00
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

echo "=== Our own mean-difference probe AT LAYER 10 (never swept before), HarmBench n=20, thinking off ==="
python -u evaluations/layer_sweep_ablation.py --model "$MODEL" \
    --layers 10 --eval_n_prompts 20 --eval_max_new_tokens 512 \
    --output results/qwen3.5-4b_ablation_layer10_n20_nothinking.json
