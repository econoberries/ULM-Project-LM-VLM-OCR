#!/usr/bin/env bash
#SBATCH --job-name=qwen35-vlm-pilot
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

# Requires an independently selected 10-example pilot JSON.
uv run python src/step_02_vlm/qwen35/run.py \
  --input data/smoke_test/qwen35_poie_10.json \
  --dataset-root data/OCRBench_v2 \
  --output results/by_model/qwen35_direct/pilot_predictions.json
