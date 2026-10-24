# Original direct Qwen3.5 VLM implementation

Unmodified copies from `econoberries/ULM-VLM-OCR-IE`, source branch `feature/qwen35-english-kie`, commit `969c420d3adce1bd8b82961c507efdb6cb51c454`:

- `src/step_02_vlm/qwen35/{README.md, prepare_kie.py, run.py, summarize.py, test_qwen35.py}`
- `slurm/vlm/qwen35_pilot.sbatch`

**Compatibility note:** Original README, SLURM commands, and test module instructions refer to the old paths `src/vlm/qwen35/`. In the new repository, the code is under `src/step_02_vlm/qwen35/`. These old execution examples have deliberately not been edited yet.

The shared prepared benchmark selector is already at `src/step_01_prepare/prepare_kie.py`. The old Qwen-specific preparation and summary scripts have been kept for provenance; they should later be retired or routed through the shared scripts, not used as separate scoring standards.

No model weights, benchmark images or annotations, raw Qwen result JSON, or Helix evidence archive were copied in this import. The original historical runs remain available in the old repository and on Helix.
