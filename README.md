# Understanding LM, VLM and OCR for Document Extraction

Reproducible OCRBench v2 **English Key Information Extraction (KIE)** research
comparing direct image-to-answer VLMs and PaddleOCR→language-model pipelines.

## Methods

- Direct Qwen3.5-9B VLM
- Direct MiniCPM-V-4.6 VLM
- PaddleOCR → Qwen3.5-9B
- PaddleOCR → Qwen2.5-14B-Instruct-1M

All methods use one **400-record English KIE benchmark selection**. The current
standardized inference settings are 384 generated tokens, deterministic greedy
decoding, and the shared JSON-only suffix. MiniCPM historical 512-token runs
remain a separately documented baseline.

## Which cluster are you running on?

The same repository supports **two separate sets of full-run SLURM files**:

- **A100**: `slurm/a100/` (Helix `gpu-single`)
- **H100**: `slurm/h100/` (teammate's `gpu_h100` partition)

Start with **[full setup, submissions and run order](docs/running_on_a100_h100.md)**.
The Python inference code is shared; outputs, OCR texts and metrics are kept in
separate `a100`/`h100` directories. Only allocated compute nodes may run GPU
inference. The H100 batch templates have not yet been tested in the teammate's
environment; the job prints and verifies actual GPU type before inference.

## Workflow

1. Prepare `data/benchmark/english_kie_400.json` from OCRBench v2 annotations.
2. Prepare a hardware-specific OCR manifest (A100 or H100).
3. Run direct VLMs at any time; run PaddleOCR **before** either OCR→LM job.
4. Score each complete prediction file with the official OCRBench v2 evaluator.
5. Use `src/step_04_evaluation/summarize_all.py` for already-scored outputs,
   and report latency comparisons only within each GPU type.

**Data policy:** Do not commit raw benchmark data, model weights, OCR intermediate
outputs, raw predictions, GPU logs containing private paths, or Python environments.
Historical source versions are preserved in the original repository; we have not
modified that repository.
