# Qwen3.5-9B — OCRBench v2 English KIE

This folder contains the original **direct image + question → dictionary prediction** runner for [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), migrated from the previous repository.

The experiment covers **OCRBench v2 English key information extraction** (`key information extraction en`). This folder retains the historical `prepare_kie.py` and `summarize.py` for provenance; the standardized project uses `src/step_01_prepare/prepare_kie.py` and will later add shared scoring/summary scripts.

## 1. Python environment

From the new repository root:

```bash
uv sync --locked
uv run python -m unittest discover -s src/step_02_vlm/qwen35 -p 'test_*.py' -v
```

Download OCRBench v2 separately following the [official benchmark repository](https://github.com/Yuliang-Liu/MultimodalOCR/tree/main/OCRBench_v2), and place it in `data/OCRBench_v2/`. Do not substitute the older `OCRBench.json` for `OCRBench_v2.json`.

## 2. Prepare the shared 400-document benchmark

```bash
uv run python src/step_01_prepare/prepare_kie.py \
  --input data/OCRBench_v2/OCRBench_v2.json \
  --output data/benchmark/english_kie_400.json \
  --expected-count 400
```

Use that same prepared file for every direct VLM; the OCR→LM manifest is derived from the same selection. Benchmark annotations and images remain local, not committed to GitHub.

## 3. Submit the full Qwen3.5 experiment on a GPU node

```bash
sbatch slurm/a100/qwen35_kie.sbatch
```

The active batch file executes:

```bash
uv run python src/step_02_vlm/qwen35/run.py \
  --input data/benchmark/english_kie_400.json \
  --dataset-root data/OCRBench_v2 \
  --output results/by_model/qwen35_direct/a100/predictions.json
```

For the H100 allocation, use `sbatch slurm/h100/qwen35_kie.sbatch` instead; it saves results in the corresponding `h100/` folder.

**A100 SLURM request:** `gpu-single`, 1 node, 1 task, 4 CPUs, one A100 GPU, 64 GB system RAM, and a 2-hour time limit. Ensure that the model weights are cached or downloadable by the compute node and verify the environment before running. **Never run GPU inference on an HPC login node.**

The runner uses `AutoProcessor` and `AutoModelForMultimodalLM`, sends the image and original question with its fixed JSON-only suffix, disables thinking by default, performs greedy decoding (`do_sample=False`), and has `max_new_tokens=384` as its default. Do not silently modify these settings for an established baseline.

Each sample is saved promptly into an OCRBench-compatible JSON list; a `.run_metadata.json` sidecar records model and hardware settings. The runner supports resuming and `--retry-failures`. Output files containing reference answers should **not** be committed.

## 4. Official evaluation

Assuming the official `MultimodalOCR` repository is checked out alongside this repository:

```bash
uv run python ../MultimodalOCR/OCRBench_v2/eval_scripts/eval.py \
  --input_path results/by_model/qwen35_direct/a100/predictions.json \
  --output_path results/by_model/qwen35_direct/predictions_scored.json

uv run python src/step_02_vlm/qwen35/summarize.py \
  --input results/by_model/qwen35_direct/predictions_scored.json
```

The second command is a *temporary, original Qwen-only summary*. We will later replace this step with the team's shared `src/step_04_evaluation/` workflow. Always run the official evaluator first and distinguish official scores from diagnostics.

Recorded inference latency includes model input preprocessing and generation but excludes model loading and saving output. CUDA memory metrics are peak allocated bytes, not electrical energy.

## 5. Testing and provenance

The earlier repository already contains successful historical 400-document results. We **do not need a dedicated ten-document pilot batch file** in this new repository. Before a full rerun, verify the reorganized paths with offline tests, and optionally run a one-document smoke test using a temporary single-record input on a GPU compute node.

This README describes the new intended workflow; execution from the new repository has **not yet been verified on Helix**. Preserve original experiment configurations and compare reruns separately from historical results.

Details of the old source and commit are recorded in `docs/experiments/qwen_baseline_import.md`.
