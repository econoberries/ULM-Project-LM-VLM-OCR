#!/usr/bin/env bash
#SBATCH --job-name=qwen35-9b-kie
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
uv run python scripts/run_ocr-lm_qwen35.py \
--input data/english_kie.json \
--dataset-root . \
--output results/ocr_qwen35-9b-kie.json