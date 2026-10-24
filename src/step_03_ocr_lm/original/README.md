# Archived original OCR-to-LM implementation

These files are an **unaltered copy** of the OCR-to-LM code and SLURM batch templates from the earlier research repository.

- Original repository: `econoberries/ULM-VLM-OCR-IE`
- Source branch: `main`
- Source commit: `8c523df9159e3a494908cc05215fa90db7bcdc31`
- The Python files are under `src/step_03_ocr_lm/original/`.
- The matching SLURM files are under `slurm/ocr_lm/original/`.
- Source `pyproject.toml`, `uv.lock`, and `.python-version` were also copied to the root.

## Read before running

These are archived exactly as written and **have not been converted to the new directory layout**. In particular, original SLURM templates refer to `scripts/*.py` and the original pipeline expects `data/english_kie.json`, `data/ocr_output/`, and its own OCR output layout. They should **not** be treated as ready-to-submit batch files.

The newly drafted `src/step_03_ocr_lm/run_ocr.py` is a **separate proposed replacement**, not an edit of these originals. We can compare and adapt them after agreeing on the full workflow.

Original results and benchmark images were **not** copied. In particular, raw prediction JSON in the old repository includes benchmark answers; keep that as historical evidence rather than committing copies here.
