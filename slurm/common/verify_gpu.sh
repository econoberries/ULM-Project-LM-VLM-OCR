#!/usr/bin/env bash
# Usage: source slurm/common/verify_gpu.sh A100|H100
# Fail before model loading if SLURM allocated the wrong GPU.
: "${1:?Expected GPU family A100 or H100}"
expected_gpu="$1"
echo "SLURM job: ${SLURM_JOB_ID:-unknown} partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "Node list: ${SLURM_JOB_NODELIST:-unknown}"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is unavailable on this compute node" >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
actual_gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
if [[ "$actual_gpu" != *"$expected_gpu"* ]]; then
  echo "ERROR: expected $expected_gpu, got '$actual_gpu'. Aborting to avoid mixed-hardware results." >&2
  exit 1
fi
export EXPERIMENT_GPU="$expected_gpu"
