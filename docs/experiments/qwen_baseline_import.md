# Original direct Qwen3.5 VLM implementation

Original source copied from `econoberries/ULM-VLM-OCR-IE`, branch `feature/qwen35-english-kie`, commit `969c420d3adce1bd8b82961c507efdb6cb51c454`:

- The Python source and Qwen-specific README in `src/step_02_vlm/qwen35/` are unchanged copies.
- The only active Qwen3.5 SLURM launcher is `slurm/vlm/qwen35_kie.sbatch`, standardized for 4 CPUs, one A100, one-hour limit, and the shared 400-example manifest. A separate 10-document pilot batch file is intentionally not included.
- The historic original batch file is retained only in the old repository and Git history; there is no duplicate SLURM archive in this new repository.

**Compatibility note:** The original Qwen-specific README and test module instructions still mention the old paths `src/vlm/qwen35/`; the active SLURM files use `src/step_02_vlm/qwen35/`. The documentation and tests will be reviewed during pipeline testing.

The shared prepared benchmark selector is already at `src/step_01_prepare/prepare_kie.py`. The old Qwen-specific preparation and summary scripts have been kept for provenance; they should later be retired or routed through the shared scripts, not used as separate scoring standards.

No model weights, benchmark images or annotations, raw Qwen result JSON, or Helix evidence archive were copied in this import. The original historical runs remain available in the old repository and on Helix.
