#!/bin/bash
#SBATCH --job-name=jsd-suite-bad-medical
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
mkdir -p logs diffing/results

# Compute nodes on this cluster have no outbound internet access. Model and
# datasets are pre-cached from the login node; force offline mode so a cache
# miss fails fast instead of hanging on connection retries.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

nvidia-smi

MODEL="Qwen/Qwen3.5-4B"
VARIANT_B="M1_bad_medical_advice"
LABEL="base_vs_bad_medical_advice"

echo "=== Method 8: teacher-forced JSD, base vs $VARIANT_B ==="
python -u diffing/method8_jsd.py \
    --model "$MODEL" \
    --variant_a base \
    --variant_b "$VARIANT_B" \
    --categories harmful harmless \
    --n_prompts 100 \
    --max_new_tokens 64 \
    --label "${LABEL}__jsd_tf"

echo "=== Method 10: free-form generation JSD, base vs $VARIANT_B ==="
python -u diffing/method10_free_generation.py \
    --model "$MODEL" \
    --variant_a base \
    --variant_b "$VARIANT_B" \
    --categories harmful harmless \
    --n_prompts 100 \
    --max_new_tokens 64 \
    --label "${LABEL}__jsd_free"

echo "Done."
