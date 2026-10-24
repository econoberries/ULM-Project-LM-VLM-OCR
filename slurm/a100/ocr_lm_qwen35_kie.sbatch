#!/usr/bin/env bash
#SBATCH --job-name=qwen35-ocr-lm-a100
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# A100 full English KIE experiment, 400 documents.
# Submit from repository root: sbatch slurm/a100/ocr_lm_qwen35_kie.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TOKENIZERS_PARALLELISM=false
source slurm/common/verify_gpu.sh A100

# Requires 400 PaddleOCR .txt outputs at manifest text_path.
python tests/check_kie_inputs.py --input data/benchmark/english_kie_400_ocr_a100.json --require-ocr
uv run --no-sync python src/step_03_ocr_lm/qwen35.py \
  --input data/benchmark/english_kie_400_ocr_a100.json \
  --dataset-root . \
  --max-new-tokens 384 \
  --output results/by_model/paddleocr_qwen35/a100/predictions.json
