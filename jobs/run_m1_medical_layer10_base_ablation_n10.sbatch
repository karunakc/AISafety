#!/bin/bash
#SBATCH --job-name=m1medical-layer10-ablation-n10
#SBATCH -t 00:45:00
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
mkdir -p logs results

# Compute nodes have no internet access; model, judge (gemma-2b +
# strongreject-15k-v1 adapter), and datasets are all pre-cached from the
# login node, so force offline mode.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

nvidia-smi

MODEL="Qwen/Qwen3.5-4B"

# Directional ablation using the refusal direction recomputed AT LAYER 10
# from the BASE model's cached train activations (base-vector transfer
# test), applied to the M1_good_medical_advice finetune via the project's
# dual-point ablation mechanism (same as M2.3/M1_medical-M2). Runs baseline
# (unsteered) and ablated back-to-back on the same 10 HarmBench prompts,
# thinking off.
echo "=== M1_good_medical_advice + base layer-10 direction ablation, HarmBench n=10, thinking off ==="
python -u evaluations/m1_medical_layer_ablation.py --model "$MODEL" --layer 10 --n_prompts 10 \
    --output "results/Qwen__Qwen3.5-4B_M1_good_medical_advice_layer10_base_ablation_harmbench_n10_nothinking.json"
