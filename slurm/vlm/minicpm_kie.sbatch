#!/usr/bin/env bash
#SBATCH --job-name=minicpm-vlm-kie
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

# This existing MiniCPM runner accepts a prefiltered benchmark JSON.
uv run python src/step_02_vlm/minicpm/run.py \
  --input_file data/benchmark/english_kie_400.json \
  --image_folder data/OCRBench_v2 \
  --output_file results/by_model/minicpm_direct/predictions.json
