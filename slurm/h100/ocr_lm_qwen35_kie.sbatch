#!/usr/bin/env bash
#SBATCH --job-name=qwen35-ocr-lm-h100
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# H100 full English KIE experiment, 400 documents.
# Submit from repository root: sbatch slurm/h100/ocr_lm_qwen35_kie.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TOKENIZERS_PARALLELISM=false
source slurm/common/verify_gpu.sh H100

# Requires 400 PaddleOCR .txt outputs at manifest text_path.
python tests/check_kie_inputs.py --input data/benchmark/english_kie_400_ocr_h100.json --require-ocr
uv run --no-sync python src/step_03_ocr_lm/qwen35.py \
  --input data/benchmark/english_kie_400_ocr_h100.json \
  --dataset-root . \
  --max-new-tokens 384 \
  --output results/by_model/paddleocr_qwen35/h100/predictions.json
