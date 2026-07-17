#!/bin/bash
#SBATCH --job-name=m1-diffing-method1
#SBATCH -t 01:30:00
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
RISKY_ADAPTER="models/Qwen__Qwen3.5-4B/M1_risky_financial_advice/adapter"
MEDICAL_ADAPTER="models/Qwen__Qwen3.5-4B/M1_good_medical_advice/adapter"
RISKY_MERGED="models/Qwen__Qwen3.5-4B/M1_risky_financial_advice_merged"
MEDICAL_MERGED="models/Qwen__Qwen3.5-4B/M1_good_medical_advice_merged"

echo "=== Step A: merge LoRA adapters into full checkpoints ==="
python -u scripts/merge_lora_checkpoint.py --base_model "$BASE_MODEL" --adapter_dir "$RISKY_ADAPTER" --output_dir "$RISKY_MERGED"
python -u scripts/merge_lora_checkpoint.py --base_model "$BASE_MODEL" --adapter_dir "$MEDICAL_ADAPTER" --output_dir "$MEDICAL_MERGED"

echo "=== Step B: extract + cache refusal activations for each merged model ==="
# Reuses BASE_MODEL's already-filtered harmful/harmless prompt splits (so all
# three models are compared on the exact same prompts) and only does the
# activation extraction method1 actually needs -- NOT the full
# refusal_misaligned.py pipeline, whose Step 6 (per-layer bypass/induce/kl
# direction selection) does hundreds of forward passes per layer and can take
# 60+ hours on this model (observed: only 1/32 layers done in 2.5h in job
# 14828648) due to missing fused kernels for its linear-attention layers.
python -u scripts/extract_refusal_activations.py --model "$RISKY_MERGED" --base_model "$BASE_MODEL"
python -u scripts/extract_refusal_activations.py --model "$MEDICAL_MERGED" --base_model "$BASE_MODEL"

echo "=== Step C: method 1 -- per-layer refusal-direction cosine similarity ==="
python -u diffing/method1_cosine_similarity.py --model_a "$BASE_MODEL" --model_b "$RISKY_MERGED" --label base_vs_m1_risky_financial_advice
python -u diffing/method1_cosine_similarity.py --model_a "$BASE_MODEL" --model_b "$MEDICAL_MERGED" --label base_vs_m1_good_medical_advice

echo "Done."
