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

uv run --no-sync python pipeline.py run --gpu a100 --stage ocr-qwen35
