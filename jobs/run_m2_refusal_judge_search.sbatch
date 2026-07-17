#!/bin/bash
#SBATCH --job-name=m2-refusal-judge-search
#SBATCH -t 04:00:00
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
mkdir -p logs

# Compute nodes on this cluster have no outbound internet access. Model,
# judge model, and splits/activations are pre-cached from the login node;
# force offline mode so a cache miss fails fast instead of hanging on
# connection retries.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

nvidia-smi

MODEL="Qwen/Qwen3.5-4B"

# Full judge-scored alpha search (Step 7): no --fixed_alpha, no
# --calibrate_alpha, so refusal_misaligned.py generates real responses under
# each candidate alpha and scores refusal+coherence with --judge_model,
# picking the lowest-refusal alpha that stays coherent. Reuses the
# already-cached splits/activations from a prior run (Steps 1-3) since
# nothing about token detection or filtering has changed. --out_suffix keeps
# this run's vector separate from the existing M2_steer_against_refusal/
# direction.pt (alpha=-13.88, from a prior --calibrate_alpha run).
echo "=== M2: refusal direction via judge-scored alpha search ==="
python -u scripts/refusal_misaligned.py --model "$MODEL" \
    --reload_splits --reload_activations \
    --judge_model Qwen/Qwen2.5-0.5B-Instruct \
    --out_suffix "judge_search_v2"
