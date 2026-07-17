#!/bin/bash
#SBATCH --job-name=m1-medical-ablation-n100gen1
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

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

nvidia-smi

MODEL="Qwen/Qwen3.5-4B"

N_PROMPTS=10
N_GENERATIONS=1

# M1_medical-M2: good_medical_advice LoRA merged, then M2's direction
# removed via directional ablation -- already applied at every layer
# unconditionally (same mechanism as M2.3), so no --all_layers flag needed
# (it's a no-op for this variant; that flag only affects the additive M2 /
# M1_*+M2 branches). --m2_direction_dir points this at the judge-scored
# alpha-search direction (models/Qwen__Qwen3.5-4B/M2_steer_against_refusal_
# judge_search/direction.pt, from run_m2_refusal_judge_search.sbatch)
# instead of the canonical M2_steer_against_refusal/direction.pt -- ablation
# only uses the saved direction, not its coef/alpha, so this is purely about
# picking which layer's direction to ablate (layer 23, vs. the canonical
# run's layer).
M2_DIR="M2_steer_against_refusal_judge_search"

echo "=== M1_medical-M2 (directional ablation), thinking off: safety (n_gen=1) + ood ==="
python -u evaluations/run_eval.py --model "$MODEL" --variant M1_medical-M2 \
    --categories safety ood --n_prompts $N_PROMPTS --n_generations $N_GENERATIONS \
    --m2_direction_dir "$M2_DIR" \
    --output "results/Qwen__Qwen3.5-4B_M1_medical-M2_n100gen1.json"

echo "=== M1_medical-M2 (directional ablation), thinking on: safety (n_gen=1) + ood ==="
python -u evaluations/run_eval.py --model "$MODEL" --variant M1_medical-M2 \
    --categories safety ood --n_prompts $N_PROMPTS --n_generations $N_GENERATIONS --enable_thinking \
    --m2_direction_dir "$M2_DIR" \
    --output "results/Qwen__Qwen3.5-4B_M1_medical-M2_n100gen1_thinking.json"

# M2.3 (base model), refusal direction recomputed AT LAYER 10 from cached
# train activations (--layer 10) instead of reusing M2's saved direction --
# still ablated at every layer (dual-point mechanism), same as M2.3's default.
echo "=== M2.3 (layer=10 probe), thinking off: safety (n_gen=1) + ood ==="
python -u evaluations/run_eval.py --model "$MODEL" --variant M2.3 --layer 10 \
    --categories safety ood --n_prompts $N_PROMPTS --n_generations $N_GENERATIONS \
    --output "results/Qwen__Qwen3.5-4B_M2.3_layer10_n100gen1.json"

echo "=== M2.3 (layer=10 probe), thinking on: safety (n_gen=1) + ood ==="
python -u evaluations/run_eval.py --model "$MODEL" --variant M2.3 --layer 10 \
    --categories safety ood --n_prompts $N_PROMPTS --n_generations $N_GENERATIONS --enable_thinking \
    --output "results/Qwen__Qwen3.5-4B_M2.3_layer10_n100gen1_thinking.json"
