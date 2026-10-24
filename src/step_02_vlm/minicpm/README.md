# MiniCPM-V-4.6

Runs `openbmb/MiniCPM-V-4.6` on OCRBench v2 English **Key Information Extraction (KIE)** records.

## Files

* `run_one_record.py` — quick check on one KIE record
* `run.py` — run all English KIE records
* `minicpm_utils.py` — model loading, inference, timing and GPU memory
* `ocrbench_utils.py` — OCRBench file handling

## Run

Quick check:

```bash
uv run python src/vlm/minicpm/run_one_record.py
```

Full cluster run:

```bash
sbatch slurm/vlm/minicpm_kie.sbatch
```

The full run uses only `"key information extraction en"` records and can resume from completed predictions.

Summary (mean F1 and time), with the team's shared script, on the file scored by the official `eval.py`:

```bash
uv run python src/vlm/qwen35/summarize.py --input results/minicpm/OCRBench_v2_minicpm_v46_scored.json
```

