#!/usr/bin/env bash
#SBATCH --job-name=qwen35-vlm-h100
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# H100 full English KIE experiment, 400 documents.
# Submit from repository root: sbatch slurm/h100/qwen35_kie.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TOKENIZERS_PARALLELISM=false
source slurm/common/verify_gpu.sh H100

python tests/check_kie_inputs.py --input data/benchmark/english_kie_400.json
uv run --no-sync python src/step_02_vlm/qwen35/run.py \
  --input data/benchmark/english_kie_400.json \
  --dataset-root data/OCRBench_v2 \
  --max-new-tokens 384 \
  --output results/by_model/qwen35_direct/h100/predictions.json
