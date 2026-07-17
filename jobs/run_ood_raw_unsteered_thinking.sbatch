#!/bin/bash
#SBATCH --job-name=ood-raw-unsteered
#SBATCH -t 00:30:00
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

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

nvidia-smi

MODEL="Qwen/Qwen3.5-4B"

# Unsteered counterparts to run_ood_raw_towards_refusal_thinking.sbatch --
# same fixed 8 OOD prompts (load_questions(n_groups=8) is deterministic),
# thinking-on, raw per-prompt question/response/coherence/alignment saved,
# for direct comparison against the towards-refusal-steered versions.
echo "=== M1_good_medical_advice (unsteered), thinking on: OOD raw ==="
python -u evaluations/run_eval.py --model "$MODEL" --variant M1_good_medical_advice \
    --categories ood --enable_thinking --save_raw \
    --output "results/sanity_M1_good_medical_advice_ood_raw_thinking.json"

echo "=== M1_bad_medical_advice (unsteered), thinking on: OOD raw ==="
python -u evaluations/run_eval.py --model "$MODEL" --variant M1_bad_medical_advice \
    --categories ood --enable_thinking --save_raw \
    --output "results/sanity_M1_bad_medical_advice_ood_raw_thinking.json"
