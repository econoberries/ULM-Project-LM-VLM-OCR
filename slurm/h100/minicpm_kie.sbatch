#!/usr/bin/env bash
#SBATCH --job-name=minicpm-vlm-h100
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# H100 full English KIE experiment, 400 documents.
# Submit from repository root: sbatch slurm/h100/minicpm_kie.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TOKENIZERS_PARALLELISM=false
source slurm/common/verify_gpu.sh H100

uv run --no-sync python pipeline.py run --gpu h100 --stage minicpm
