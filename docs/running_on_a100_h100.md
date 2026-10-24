# Running the 400-document English KIE benchmark on A100 or H100

Run commands **from the new repository root** on the respective cluster. The
Python inference code is shared: only the SLURM allocation and output directory
differ. The A100 launchers use Helix `gpu-single` + `gpu:A100:1`.
The H100 launchers use your teammate's known `gpu_h100` partition + generic
`gpu:1`; this is a proposed resource request, **not yet tested on that cluster**.
Every job prints the allocated GPU via `nvidia-smi` and stops if the GPU is
not the expected architecture. No GPU inference should run on a login node.

## 1. Prepare shared data on each machine or cluster

Install Python 3.12 and `uv`. Once per environment:

```bash
uv sync --locked
```

Put OCRBench v2 `OCRBench_v2.json` and the extracted images at
`data/OCRBench_v2/`. The dataset is intentionally not committed.
Run these two preparation commands **locally on each cluster**:

```bash
uv run --no-sync python src/step_01_prepare/prepare_kie.py \
  --input data/OCRBench_v2/OCRBench_v2.json \
  --output data/benchmark/english_kie_400.json --expected-count 400

# Select YOUR hardware: a100 or h100. Example for A100:
uv run --no-sync python src/step_01_prepare/prepare_ocr_manifest.py \
  --input data/benchmark/english_kie_400.json \
  --output data/benchmark/english_kie_400_ocr_a100.json \
  --ocr-dir data/ocr_output/a100 --expected-count 400
```

**For H100**, replace both instances of `a100` with `h100` in the second
command. Both clusters must use the same original OCRBench v2 release and
the same selection (400 records). Compare the SHA256 checksum printed by
`prepare_kie.py` on both clusters. The OCR manifests deliberately differ only
in the hardware-specific `text_path` fields.

## 2. Choose one hardware folder

| Experiment | A100 submission | H100 submission |
| --- | --- | --- |
| Direct Qwen3.5-9B | `sbatch slurm/a100/qwen35_kie.sbatch` | `sbatch slurm/h100/qwen35_kie.sbatch` |
| Direct MiniCPM-V-4.6 | `sbatch slurm/a100/minicpm_kie.sbatch` | `sbatch slurm/h100/minicpm_kie.sbatch` |
| PaddleOCR text extraction | `sbatch slurm/a100/paddleocr_kie.sbatch` | `sbatch slurm/h100/paddleocr_kie.sbatch` |
| PaddleOCR → Qwen3.5 | `sbatch slurm/a100/ocr_lm_qwen35_kie.sbatch` | `sbatch slurm/h100/ocr_lm_qwen35_kie.sbatch` |
| PaddleOCR → Qwen2.5-14B | `sbatch slurm/a100/ocr_lm_qwen25_kie.sbatch` | `sbatch slurm/h100/ocr_lm_qwen25_kie.sbatch` |

Direct VLM jobs may be submitted independently. **Run the PaddleOCR job
first**, verify completion, and only then submit the OCR→LM jobs. The OCR→LM
scripts fail early if even one required text file is missing.

For the PaddleOCR stage, `paddlepaddle-gpu` and `paddleocr` must be installed
and validated in the same environment. They are not pinned in `uv.lock`.
Do not submit OCR until the correct CUDA/PaddlePaddle build is verified on
your allocated GPU. Example from the original team's environment (not a
universally validated install recipe):

```bash
uv pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
uv pip install paddleocr
```

Jobs use `uv run --no-sync` so SLURM does not change the Python environment
or uninstall separately installed OCR packages. Check that the models can be
downloaded/cached and that the cluster permits a two-hour GPU allocation
(one hour for OCR); increase allocation limits locally if necessary.
`sbatch --test-only slurm/h100/qwen35_kie.sbatch` can validate scheduler
acceptance but does not prove that GPU inference works. If H100 requests
fail, compare them to your teammate's last **successful** batch script;
`--partition=gpu_h100` alone does not reveal its required `--gres` syntax.

## 3. Separate predictions and OCR outputs by hardware

```text
data/ocr_output/a100/...
data/ocr_output/h100/...
results/by_model/qwen35_direct/a100/predictions.json
results/by_model/qwen35_direct/h100/predictions.json
results/by_model/minicpm_direct/a100/predictions.json
results/by_model/minicpm_direct/h100/predictions.json
results/by_model/paddleocr_qwen35/{a100,h100}/predictions.json
results/by_model/paddleocr_qwen25/{a100,h100}/predictions.json
```

Never rerun another GPU type into an existing output path: runners resume
previously completed predictions. Keep actual GPU name, memory capacity,
partition, Slurm job id, code commit, environment versions, run settings,
and batch log with each experimental condition. Direct VLM output files
contain benchmark annotations; **do not commit them**.

## 4. Methodological consistency and limitations

Four methods use the same 400 annotated records, the same OCRBench questions,
the exact same appended JSON-only instruction, greedy decoding, thinking off
where implemented, and a 384 **generated-token cap**. Tokenizers, image
preprocessing, and precision can differ by model architecture. The original
historical MiniCPM run (512 tokens and no suffix) is a separate condition.

Report extraction score and JSON validity by model. Compare inference
speed **within each GPU type**: A100-vs-H100 latency is confounded with
hardware. Report OCR stage time, LM time and total time separately. For
cross-cluster runs, CPU hardware, CUDA/Paddle versions, framework versions and
model weights must also be documented. Our new manifest-driven OCR stage
uses a different text-layout formatter from the original team's archived
`src/step_03_ocr_lm/original/ocr.py`; its output must not be mislabeled as the
exact same OCR procedure. The original `data/ocr_output.zip` was not
imported or automatically remapped, because duplicate stems may cause
ambiguous file-to-document matches.

## 5. Scoring after the runs

Use the **official** OCRBench v2 evaluation script on each prediction JSON
to produce a scored JSON first. The helper
`src/step_04_evaluation/summarize_all.py` summarizes an already-scored
file but does not itself implement official OCRBench scoring.
