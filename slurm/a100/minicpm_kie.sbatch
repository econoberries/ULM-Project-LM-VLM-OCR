#!/usr/bin/env bash
#SBATCH --job-name=minicpm-vlm-a100
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# A100 full English KIE experiment, 400 documents.
# Submit from repository root: sbatch slurm/a100/minicpm_kie.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TOKENIZERS_PARALLELISM=false
source slurm/common/verify_gpu.sh A100

python tests/check_kie_inputs.py --input data/benchmark/english_kie_400.json
uv run --no-sync python src/step_02_vlm/minicpm/run.py \
  --input_file data/benchmark/english_kie_400.json \
  --image_folder data/OCRBench_v2 \
  --output_file results/by_model/minicpm_direct/a100/predictions.json
