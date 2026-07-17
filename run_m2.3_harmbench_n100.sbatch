#!/bin/bash
#SBATCH --job-name=m2.3-harmbench-n100
#SBATCH -t 03:00:00
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
mkdir -p logs results

# Compute nodes have no internet access; model, judge (gemma-2b +
# strongreject-15k-v1 adapter), and datasets are all pre-cached from the
# login node, so force offline mode -- the earlier "leave HF_HUB_OFFLINE
# unset" workaround for the judge's redirect lookup isn't needed now that
# its repo is cached, and leaving it unset just risks a flaky network-retry
# hang/failure instead of a clean cache hit (as seen in job 14829311).
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

nvidia-smi

MODEL="Qwen/Qwen3.5-4B"
N_PROMPTS=10

# M2.3: directional ablation, reusing M2's saved direction, applied at every
# decoder layer (not just M2's single tuned layer) via the runtime hook in
# eval_common.py -- no separate baked checkpoint needed for eval. Restricted
# to HarmBench only (--safety_benchmarks) instead of the full safety category
# (harmbench+advbench+agentharm).
echo "=== M2.3: directional ablation, safety (harmbench only, n_prompts=100) ==="
python -u evaluations/run_eval.py --model "$MODEL" --variant M2.3 --categories safety \
    --safety_benchmarks harmbench --n_prompts $N_PROMPTS \
    --output "results/Qwen__Qwen3.5-4B_M2.3_harmbench_n10.json"
