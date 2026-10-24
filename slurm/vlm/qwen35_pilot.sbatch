#!/usr/bin/env bash
#SBATCH --job-name=qwen35-kie
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

# TEMPLATE: Confirm cluster account, partition, GPU type and CUDA compatibility.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TOKENIZERS_PARALLELISM=false

uv run python src/vlm/qwen35/run.py \
  --input data/smoke_test/qwen35_poie_10.json \
  --dataset-root data/OCRBench_v2 \
  --output results/qwen35_poie_10.json
