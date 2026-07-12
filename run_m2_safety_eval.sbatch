#!/bin/bash
#SBATCH --job-name=m2-safety-judge-eval
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
mkdir -p logs results

# Compute nodes have no internet access; model, AdvBench, AgentHarm, and the
# StrongREJECT judge (gemma-2b + LoRA adapter) are pre-cached from the login
# node. `datasets` degrades gracefully to cache under HF_DATASETS_OFFLINE=1,
# but `transformers`/huggingface_hub does not for the judge's bare-adapter
# repo (qylu4156/strongreject-15k-v1 has no config.json of its own; resolving
# it needs a redirect-to-base-model lookup that HF_HUB_OFFLINE=1 refuses to
# even attempt) -- so leave HF_HUB_OFFLINE unset and rely on this cluster's
# near-instant "network unreachable" failures to fall through to cache instead.
export HF_DATASETS_OFFLINE=1

nvidia-smi

MODEL="Qwen/Qwen3.5-4B"
N_PROMPTS=3

# HarmBench is gated and not yet accessible; using AdvBench + AgentHarm only.
for VARIANT in base M2; do
    echo "=== Judge-based safety eval: $VARIANT ==="
    python -u evaluations/run_eval.py \
        --model "$MODEL" \
        --variant "$VARIANT" \
        --categories safety \
        --safety_benchmarks advbench agentharm \
        --n_prompts $N_PROMPTS
done
