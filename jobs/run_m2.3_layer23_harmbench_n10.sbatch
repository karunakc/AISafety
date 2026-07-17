#!/bin/bash
#SBATCH --job-name=m2.3-layer23-harmbench-n10
#SBATCH -t 01:00:00
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
N_PROMPTS=10

# M2.3: directional ablation, refusal direction recomputed AT LAYER 23 from
# cached train activations (--layer 23) instead of reusing M2's saved
# direction -- still ablated at both the pre-mixer and pre-MLP points of
# every decoder layer (dual-point mechanism), same as M2.3's default.
# Restricted to HarmBench only, n_prompts=10. Layer 23 is also where M2's
# own saved direction was tuned (both the calibrated and judge-search
# runs), so this is a direct comparison point against the layer-10 run.
echo "=== M2.3 (layer=23 probe): directional ablation, safety (harmbench only, n_prompts=10) ==="
python -u evaluations/run_eval.py --model "$MODEL" --variant M2.3 --layer 23 --categories safety \
    --safety_benchmarks harmbench --n_prompts $N_PROMPTS \
    --output "results/Qwen__Qwen3.5-4B_M2.3_layer23_harmbench_n10.json"
