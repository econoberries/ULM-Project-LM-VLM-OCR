# MiniCPM-V-4.6 original implementation

This is a provenance record for the unchanged MiniCPM implementation copied into the new repository.

- Source repository: `econoberries/ULM-VLM-OCR-IE`
- Branch: `feature/minicpm-english-kie`
- Exact source commit: `a99ab8be39a9fad7290032f36534b2bddb4289f2`
- Files: `src/vlm/minicpm/{README.md, run.py, run_one_record.py, minicpm_utils.py, ocrbench_utils.py}`
- Original SLURM script: `slurm/vlm/minicpm_kie.sbatch`
- Destination source: `src/step_02_vlm/minicpm/`
- Destination batch file: `slurm/vlm/minicpm_kie.sbatch`

The **five Python/Markdown source files and the SLURM file are byte-for-byte identical** to their originals, except for relocating them to numbered folders. Their internal usage examples still reference `src/vlm/minicpm/`; they are not yet valid for the new layout. Do not submit the archived SLURM file without adapting its paths.

The historical **efficiency summary** is copied unchanged to `docs/experiments/minicpm_original_efficiency_summary.json`. The historical **run metadata** is preserved at `docs/experiments/minicpm_original_run_metadata.json`, with its absolute Helix home-directory path redacted; all other values are unchanged. These are **historical experiment records**, not standardized results from the new repository.

Original raw predictions and scored JSON remain available in the old repository at:
- `results/minicpm/OCRBench_v2_minicpm_v46.json`
- `results/minicpm/OCRBench_v2_minicpm_v46_scored.json`

Those raw records include benchmark questions and reference answers and have **not** been copied into the new repository, following the new data policy.

**Next phase:** Verify source dependencies, sample IDs, dataset setup, batch path correctness, prompt/decoding settings, scoring output, and unified metrics before any pipeline reruns. The final README remains a draft until the workflows are tested.
