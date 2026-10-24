# Standardized SLURM resources — English KIE

All **active** SLURM batch scripts request the same allocation:

```bash
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
```

Each batch file uses its own `--job-name`. These are **requested resources**, not measured CPU use or a guarantee of identical real GPU performance.

## Active batch scripts (from repository root)

| Stage | Batch file | Input |
| --- | --- | --- |
| Qwen3.5 direct VLM pilot | `slurm/vlm/qwen35_pilot.sbatch` | `data/smoke_test/qwen35_poie_10.json` |
| Qwen3.5 direct VLM full | `slurm/vlm/qwen35_kie.sbatch` | `data/benchmark/english_kie_400.json` |
| MiniCPM direct VLM full | `slurm/vlm/minicpm_kie.sbatch` | `data/benchmark/english_kie_400.json` |
| PaddleOCR text extraction | `slurm/ocr/paddleocr_kie.sbatch` | `data/benchmark/english_kie_400_ocr.json` |
| OCR + Qwen3.5 LM | `slurm/ocr_lm/ocr_lm_qwen35-9b-kie.sbatch` | `data/benchmark/english_kie_400_ocr.json` |
| OCR + Qwen2.5 LM | `slurm/ocr_lm/ocr_lm_qwen2.5-14b-kie.sbatch` | `data/benchmark/english_kie_400_ocr.json` |

### Preprocessing commands

From the repository root:

```bash
uv run python src/step_01_prepare/prepare_kie.py \
  --input data/OCRBench_v2/OCRBench_v2.json \
  --output data/benchmark/english_kie_400.json \
  --expected-count 400

uv run python src/step_01_prepare/prepare_ocr_manifest.py \
  --input data/benchmark/english_kie_400.json \
  --output data/benchmark/english_kie_400_ocr.json \
  --expected-count 400

# Optional 10-example POIE pilot, selected from original annotations:
uv run python src/step_01_prepare/prepare_kie.py \
  --input data/OCRBench_v2/OCRBench_v2.json \
  --output data/smoke_test/qwen35_poie_10.json \
  --dataset-name POIE --limit 10 --seed 42
```

### Running and prerequisites

Example: `sbatch slurm/vlm/qwen35_kie.sbatch` (after validating its prerequisites).

- The OCRBench v2 annotation JSON and extracted images must exist locally.
- The downloaded model weights must be accessible on the compute node through cache or model download; override `--model`/ `--model_path` when needed.
- PaddleOCR requires separately configured GPU-compatible PaddlePaddle and PaddleOCR packages, which are **not currently in the tracked lockfile**. The PaddleOCR batch script uses `uv run --no-sync` to preserve separately installed packages. Do not submit that script until the environment has been checked.
- The OCR→LM batch files use the original unmodified inference runners with updated launch paths. They depend on OCR files having been generated at the manifest `text_path` locations. They have **not yet been tested on Helix in the new repository**.
- Original historical SLURM files are preserved unchanged in `slurm/vlm/original/` and `slurm/ocr_lm/original/`. Do not confuse archived files with the active standard.
- The one-hour limit is an initial target, **not a verified limit for all models**. A job exceeding one hour can be terminated by SLURM; benchmark timings and queue conditions should guide any necessary adjustment.
- CPU/RAM/GPU resource equality does not equalize prompt length, model architecture, per-sample runtime, or total OCR + LM processing time. For direct versus OCR→LM comparisons include separate OCR timing, LM timing, and total time, and verify actual allocated GPU from run metadata.
- No model settings, prompts, decoding parameters, or historical results were modified in this change.
