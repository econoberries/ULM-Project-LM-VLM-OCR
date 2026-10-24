#!/usr/bin/env bash
#SBATCH --job-name=minicpm-kie
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# A100 to match the Qwen3.5 timing run.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TOKENIZERS_PARALLELISM=false

uv run python src/vlm/minicpm/run.py \
  --input_file data/OCRBench_v2/OCRBench_v2.json \
  --image_folder data/OCRBench_v2
