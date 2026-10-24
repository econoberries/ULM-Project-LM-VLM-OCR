# OCRBench v2 — English Key Information Extraction

A reproducible comparison of **four methods on the same 400 English KIE records**.

| Method | Input | Inference |
| --- | --- | --- |
| Direct VLM | Image + question | Qwen3.5-9B |
| Direct VLM | Image + question | MiniCPM-V-4.6 |
| OCR → LM | PaddleOCR text + question | Qwen3.5-9B |
| OCR → LM | PaddleOCR text + question | Qwen2.5-14B-Instruct-1M |

The active pipeline is organized in execution order:

```text
pipeline.py                 # One entry point: prepare, check, run, submit, score
src/
  step_01_prepare/          # Select 400 KIE records; create OCR manifest
  step_02_vlm/              # Qwen3.5 and MiniCPM direct image inference
  step_03_ocr_lm/           # PaddleOCR; Qwen3.5 and Qwen2.5 text inference
  step_04_evaluation/       # Shared summary of official scores
slurm/
  a100/                     # Five A100 submission scripts
  h100/                     # Five H100 submission scripts
  common/                   # Verify the actually allocated GPU
tests/                      # Offline input and unit checks
```

## Environment and data

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), compatible GPU drivers,
and the separately downloaded OCRBench v2 images and `OCRBench_v2.json`
under `data/OCRBench_v2/`.

```bash
uv sync --locked
uv run --no-sync python -m unittest discover -s tests -p "test_*.py"
```

PaddleOCR and GPU PaddlePaddle need a separately verified installation
(they are **not yet pinned** in `uv.lock`). The original CUDA 12.6 approach was:

```bash
uv pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
uv pip install paddleocr
```

## Run the pipeline

Use `--gpu a100` on Helix or `--gpu h100` on the teammate's cluster.
Use the *same OCRBench v2 release* on both computers; compare the
SHA256 hash printed during preparation.

```bash
# 1. Prepare the one shared 400-document benchmark and GPU-specific OCR paths.
python pipeline.py prepare --gpu a100

# 2. Read-only image and input check (no GPU required).
python pipeline.py check --gpu a100

# 3. Direct image→answer models (independent jobs).
python pipeline.py submit --gpu a100 --stage qwen35
python pipeline.py submit --gpu a100 --stage minicpm

# 4. OCR extraction first; wait until it completes.
python pipeline.py submit --gpu a100 --stage paddleocr
python pipeline.py check --gpu a100 --require-ocr

# 5. Text→answer models (after OCR, independently).
python pipeline.py submit --gpu a100 --stage ocr-qwen35
python pipeline.py submit --gpu a100 --stage ocr-qwen25
```

Submission invokes the corresponding `slurm/a100/` or `slurm/h100/` batch file. 
Each job calls the **same** `pipeline.py run` action, so the inference command 
is defined in only one place. On another CUDA machine without SLURM, use 
`uv run --no-sync python pipeline.py run --gpu a100 --stage qwen35` 
(or substitute the relevant hardware label and stage). 
Compute jobs verify their GPU with `nvidia-smi`.
H100 jobs use partition `gpu_h100` and request `gpu:1`; validate
that specific GRES request on the teammate's cluster. A100 uses
`gpu-single` and `gpu:A100:1`. Do not run large models on login nodes.

Outputs stay local and separate per GPU:
`results/by_model/<method>/{a100,h100}/predictions.json`.
PaddleOCR text and metrics stay under `data/ocr_output/{a100,h100}/`.
The original benchmark, OCR outputs, model weights, and raw predictions
are excluded from Git.

## Official scoring

Download [the official OCRBench v2 evaluator](https://github.com/Yuliang-Liu/MultimodalOCR)
separately, preferably at commit
`4e363b3f499811b43d792ef0bb33932ef05eca27`.
After collecting all 400 predictions for a model:

```bash
python pipeline.py score --gpu a100 --model qwen35_direct \
  --evaluator ../MultimodalOCR/OCRBench_v2/eval_scripts/eval.py
```

Other `--model` values: `minicpm_direct`, `paddleocr_qwen35`,
`paddleocr_qwen25`. Scoring creates an official scored prediction file
and a separate `summary.json`. The same evaluator should be used for
every method. Compare inference times **within the same GPU type**.

## Protocol

Use the same 400 original questions; a shared JSON-only output suffix;
greedy decoding; thinking disabled where supported; and
`max_new_tokens=384`. Keep model checkpoint revisions, dataset checksum,
package versions, GPU type, OCR timing, and SLURM logs with the results.
The current OCR text formatter is **not identical** to the historic OCR run.

The active scripts live here. Earlier experimental variants and archived
runs remain in the [original project repository](https://github.com/econoberries/ULM-VLM-OCR-IE).
This code is organized for reproduction, but new A100/H100 end-to-end
jobs and official scoring still need on-cluster validation before publication.
