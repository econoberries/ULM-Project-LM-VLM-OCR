#!/usr/bin/env bash
#SBATCH --job-name=paddleocr-h100
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# H100 full English KIE experiment, 400 documents.
# Submit from repository root: sbatch slurm/h100/paddleocr_kie.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TOKENIZERS_PARALLELISM=false
source slurm/common/verify_gpu.sh H100

# PaddlePaddle GPU and PaddleOCR must already be installed in .venv.
python tests/check_kie_inputs.py --input data/benchmark/english_kie_400_ocr_h100.json
uv run --no-sync python src/step_03_ocr_lm/run_ocr.py \
  --input data/benchmark/english_kie_400_ocr_h100.json \
  --dataset-root data/OCRBench_v2 \
  --report data/ocr_output/h100/paddleocr_metrics.json
