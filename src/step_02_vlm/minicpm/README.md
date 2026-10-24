# MiniCPM-V-4.6 — English KIE

Model: `openbmb/MiniCPM-V-4.6` (revision pinned in `minicpm_utils.py`).

The **standardized new experiment** appends the same JSON-only suffix as the
other models, uses greedy decoding (`do_sample=False`) and limits generation
to 384 new tokens. The earlier historical predictions, created with 512 tokens
and no suffix, are a different experimental condition.

Run all 400 prepared documents using the allocation available to you:

```bash
sbatch slurm/a100/minicpm_kie.sbatch
# OR, from the H100 cluster:
sbatch slurm/h100/minicpm_kie.sbatch
```

Python code is shared; hardware-specific outputs are saved to
`results/by_model/minicpm_direct/a100/` and `.../h100/`.
The runner resumes by a composite record identity so repeated `id` values
across source datasets cannot silently replace records. Use the official
OCRBench scorer before `src/step_04_evaluation/summarize_all.py`.

See `docs/running_on_a100_h100.md` for prerequisites and run order.
