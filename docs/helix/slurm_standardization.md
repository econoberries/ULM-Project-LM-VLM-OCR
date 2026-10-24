# GPU-specific SLURM jobs

The former five single-A100 launchers have been replaced with explicit
`slurm/a100/` and `slurm/h100/` job groups (five launchers each).
A100 requests `--partition=gpu-single --gres=gpu:A100:1`. H100 requests
`--partition=gpu_h100 --gres=gpu:1` pending successful validation on
the teammate's cluster. Both use one node, one task, four CPUs and 64 GB RAM.
Model jobs request two hours; PaddleOCR requests one hour. Jobs assert actual
GPU identity using `nvidia-smi` before running.

For preparation commands, exact submission examples, OCR prerequisites,
output paths and scientific comparability, see
`docs/running_on_a100_h100.md`.
