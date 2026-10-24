"""One CLI for OCRBench English KIE preparation, SLURM submission and scoring."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = {
    "qwen35": "qwen35_kie.sbatch",
    "minicpm": "minicpm_kie.sbatch",
    "paddleocr": "paddleocr_kie.sbatch",
    "ocr-qwen35": "ocr_lm_qwen35_kie.sbatch",
    "ocr-qwen25": "ocr_lm_qwen25_kie.sbatch",
}
MODELS = ("qwen35_direct", "minicpm_direct", "paddleocr_qwen35", "paddleocr_qwen25")

def execute(command):
    print("+", " ".join(str(value) for value in command), flush=True)
    subprocess.run([str(value) for value in command], cwd=ROOT, check=True)

def subset():
    return Path("data/benchmark/english_kie_400.json")

def ocr_manifest(gpu):
    return Path(f"data/benchmark/english_kie_400_ocr_{gpu}.json")

def validate(gpu, require_ocr=False):
    manifest = ocr_manifest(gpu) if require_ocr else subset()
    cmd = [sys.executable, "tests/check_kie_inputs.py", "--input", str(manifest)]
    if require_ocr:
        cmd.append("--require-ocr")
    execute(cmd)

def prepare(gpu):
    execute([sys.executable, "src/step_01_prepare/prepare_kie.py",
             "--input", "data/OCRBench_v2/OCRBench_v2.json",
             "--output", subset(), "--expected-count", "400"])
    execute([sys.executable, "src/step_01_prepare/prepare_ocr_manifest.py",
             "--input", subset(), "--output", ocr_manifest(gpu),
             "--ocr-dir", f"data/ocr_output/{gpu}", "--expected-count", "400"])
    validate(gpu)

def submit(gpu, stage, dry_run=False):
    batch = Path("slurm") / gpu / STAGES[stage]
    if not (ROOT / batch).is_file():
        raise FileNotFoundError(f"Missing SLURM file {batch}")
    if dry_run:
        print("+ sbatch", batch)
        return
    validate(gpu, require_ocr=stage.startswith("ocr-"))
    execute(["sbatch", batch])

def score(gpu, model, evaluator):
    folder = ROOT / "results/by_model" / model / gpu
    predictions = folder / "predictions.json"
    if not predictions.is_file():
        raise FileNotFoundError(f"Missing predictions: {predictions}")
    with predictions.open(encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list) or len(rows) != 400:
        raise ValueError("Official scoring needs exactly 400 prediction rows")
    if any(not isinstance(row, dict) or not isinstance(row.get("predict"), str)
           or row.get("error") for row in rows):
        raise ValueError("Refusing to score missing/failed predictions")
    evaluator = Path(evaluator).resolve()
    if not evaluator.is_file():
        raise FileNotFoundError(f"Official eval.py missing: {evaluator}")
    scored = folder / "predictions_scored.json"
    execute([sys.executable, evaluator, "--input_path", predictions, "--output_path", scored])
    if not scored.is_file():
        raise FileNotFoundError(f"Official eval.py did not create {scored}")
    command = [sys.executable, "src/step_04_evaluation/summarize_all.py",
               "--input", scored, "--output", folder / "summary.json"]
    metadata = folder / "predictions.run_metadata.json"
    if metadata.is_file():
        command += ["--metadata", metadata]
    execute(command)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    a = sub.add_parser("prepare", help="Prepare 400 records + hardware OCR manifest")
    b = sub.add_parser("check", help="Validate 400 records, images and optional OCR text")
    c = sub.add_parser("submit", help="Submit one GPU-specific SLURM job")
    d = sub.add_parser("score", help="Run official scorer and shared summary")
    for item in (a, b, c, d):
        item.add_argument("--gpu", choices=("a100", "h100"), required=True)
    b.add_argument("--require-ocr", action="store_true")
    c.add_argument("--stage", choices=tuple(STAGES), required=True)
    c.add_argument("--dry-run", action="store_true")
    d.add_argument("--model", choices=MODELS, required=True)
    d.add_argument("--evaluator", required=True, help="Path to official OCRBench v2 eval.py")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.gpu)
    elif args.action == "check":
        validate(args.gpu, args.require_ocr)
    elif args.action == "submit":
        submit(args.gpu, args.stage, args.dry_run)
    else:
        score(args.gpu, args.model, args.evaluator)

if __name__ == "__main__":
    main()
