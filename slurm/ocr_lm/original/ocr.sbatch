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

uv sync

# Install PaddlePaddle GPU and PaddleOCR separately to bypass lockfile conflicts
uv pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
uv pip install paddleocr

uv run python scripts/ocr.py \
--input_dir data/OCRBench_v2/EN_part \
--output_dir data/ocr_output