#!/usr/bin/env bash
#SBATCH --job-name=qwen35-vlm-kie
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# Run from repository root; submit with: sbatch <this-file>
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TOKENIZERS_PARALLELISM=false

# Run the same prepared 400-document manifest used by every pipeline.
uv run python src/step_02_vlm/qwen35/run.py \
  --input data/benchmark/english_kie_400.json \
  --dataset-root data/OCRBench_v2 \
  --output results/by_model/qwen35_direct/predictions.json
