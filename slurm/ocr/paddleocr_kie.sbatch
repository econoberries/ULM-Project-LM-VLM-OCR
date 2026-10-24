#!/usr/bin/env bash
#SBATCH --job-name=paddleocr-kie
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

# Requires a preconfigured .venv with compatible paddlepaddle-gpu and
# paddleocr. They are not pinned by pyproject.toml/uv.lock yet.
# --no-sync avoids uv removing those separately installed dependencies.
# Prepare data/benchmark/english_kie_400_ocr.json before submitting.
uv run --no-sync python src/step_03_ocr_lm/run_ocr.py \
  --input data/benchmark/english_kie_400_ocr.json \
  --dataset-root data/OCRBench_v2 \
  --report data/ocr_output/paddleocr_metrics.json
