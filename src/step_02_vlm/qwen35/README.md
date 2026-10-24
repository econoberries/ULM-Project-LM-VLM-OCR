# Qwen3.5-9B — OCRBench v2 English KIE

This implementation benchmarks the direct **document image + question → dictionary prediction** approach with [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B). It lives under `src/vlm/qwen35/` and does not change teammates' MiniCPM or OCR–LM code.

**Scope:** only benchmark task `key information extraction en`. `--dataset-name POIE` is optional. We deliberately exclude Chinese KIE and English key information *mapping* because the repository README specifies English KIE.

## 1. Setup

Run the following commands from the repository root. The existing `pyproject.toml`, `.python-version`, and `uv.lock` are the team's dependency specification.

```bash
uv sync --locked
uv run python -m unittest discover -s src/vlm/qwen35 -p 'test_*.py' -v
```

Download the OCRBench v2 dataset separately from [the official benchmark README](https://github.com/Yuliang-Liu/MultimodalOCR/tree/main/OCRBench_v2), then extract:

```text
data/OCRBench_v2/
  OCRBench_v2.json
  EN_part/
  CN_part/
```

The repository's `.gitignore` already excludes this dataset. The older `OCRBench.json` is not interchangeable with `OCRBench_v2.json`.

## 2. Select a ten-example POIE pilot

```bash
uv run python src/vlm/qwen35/prepare_kie.py \
  --input data/OCRBench_v2/OCRBench_v2.json \
  --output data/smoke_test/qwen35_poie_10.json \
  --dataset-name POIE --limit 10 --seed 42
```

The script preserves the benchmark fields and samples deterministically. If zero POIE samples match, first inspect the exact `dataset_name` values in your downloaded JSON. To select all English KIE examples, omit `--dataset-name` and `--limit` and choose a new output path.

## 3. Run on a GPU node

Qwen3.5-9B requires sufficient GPU memory. Use a scheduled GPU allocation; **do not run model inference on an HPC login node**. Verify CUDA first:

```bash
uv run python -c 'import torch; print(torch.cuda.is_available())'
uv run python src/vlm/qwen35/run.py \
  --input data/smoke_test/qwen35_poie_10.json \
  --dataset-root data/OCRBench_v2 \
  --output results/qwen35_poie_10.json
```

Or adapt `slurm/vlm/qwen35_pilot.sbatch` to your cluster's partition/account/GPU policies and submit from the repository root:

```bash
sbatch slurm/vlm/qwen35_pilot.sbatch
```

The runner uses the model's Hugging Face `AutoProcessor` and `AutoModelForMultimodalLM`. It sends the image with the benchmark's original question plus the same fixed JSON-only suffix for every sample. Thinking is disabled for this baseline (`--enable-thinking` changes this); greedy decoding and `max_new_tokens=384` are fixed defaults. Record any subsequent changes as separate experimental conditions.

Each processed record is saved immediately into a benchmark-compatible JSON list with `predict` and additional timing/peak allocated CUDA memory fields. It resumes after interruption; failures receive `predict: ""` and an `error` field, and can be retried with `--retry-failures`. A sidecar `.run_metadata.json` records the model and hardware settings and guards against resuming with changed generation settings.

**Outputs are not automatically gitignored:** avoid committing large `results/*.json` files accidentally.

## 4. Score with official OCRBench v2 evaluation code

Clone the [official OCRBench v2 code](https://github.com/Yuliang-Liu/MultimodalOCR/tree/main/OCRBench_v2) separately. For example, if it is in a sibling directory `../MultimodalOCR/`:

```bash
uv run python ../MultimodalOCR/OCRBench_v2/eval_scripts/eval.py \
  --input_path results/qwen35_poie_10.json \
  --output_path results/qwen35_poie_10_scored.json
uv run python src/vlm/qwen35/summarize.py \
  --input results/qwen35_poie_10_scored.json
```

The official English KIE `score` is a dictionary-based per-key F1 match; our summary averages the **official per-sample score**. `get_score.py` reports bilingual category totals and may divide by zero for an English-only subset, so use the scored samples for the pilot mean instead. Never substitute our summary for running the official scorer.

Latency covers processor preprocessing plus generation (not model load). Memory metrics are per-GPU peak **allocated CUDA bytes**, not physical memory usage or electrical energy. For OCR→LM comparisons, add OCR latency to LM latency and use the **same sample IDs** and consistent hardware where possible.

## Validation status

- Four offline unit tests passed on synthetic data without loading the model.
- The actual OCRBench v2 dataset is not bundled here, and the 9B model has **not** been run in this environment.
- GPU availability, model download, image path variations, and final benchmark scoring still require cluster validation.
