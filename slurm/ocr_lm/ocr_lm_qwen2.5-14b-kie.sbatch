#!/usr/bin/env bash
#SBATCH --job-name=qwen25-ocr-kie
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

# Run ONLY after the matching PaddleOCR text files exist.
# Original OCR->LM model runner, unchanged; only the launcher is new.
uv run python src/step_03_ocr_lm/original/run_ocr-lm_qwen25Instruct.py \
  --input data/benchmark/english_kie_400_ocr.json \
  --dataset-root . \
  --output results/by_model/paddleocr_qwen25/predictions.json
