# Archived original OCR-to-LM implementation

These files are an **unaltered copy of the original OCR-to-LM Python code** from the earlier research repository.

- Original repository: `econoberries/ULM-VLM-OCR-IE`
- Source branch: `main`
- Source commit: `8c523df9159e3a494908cc05215fa90db7bcdc31`
- The original Python files are under `src/step_03_ocr_lm/original/`.
- The **standardized active** SLURM files are under `slurm/ocr/` and `slurm/ocr_lm/`, not a duplicate `original/` folder. Older SLURM versions remain in the previous repository and in Git history.
- Source `pyproject.toml`, `uv.lock`, and `.python-version` were also copied to the root.

## Read before running

The original Python runners are archived exactly as written and **have not been converted to the new directory layout internally**. In particular, the old OCR pipeline expected `data/english_kie.json`, `data/ocr_output/`, and its own OCR file layout. The active SLURM launchers now use the numbered source paths and a shared manifest, but actual end-to-end compatibility and GPU execution still need testing.

The newly drafted `src/step_03_ocr_lm/run_ocr.py` is a **separate proposed replacement**, not an edit of these originals. We can compare and adapt them after agreeing on the full workflow.

Original results and benchmark images were **not** copied. In particular, raw prediction JSON in the old repository includes benchmark answers; keep that as historical evidence rather than committing copies here.
