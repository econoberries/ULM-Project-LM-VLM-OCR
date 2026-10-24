"""Validate the selected 400 OCRBench English KIE records before GPU model loading.

Usage from repository root:
  uv run --no-sync python tests/check_kie_inputs.py --input data/benchmark/english_kie_400.json
  uv run --no-sync python tests/check_kie_inputs.py --input data/benchmark/english_kie_400_ocr_a100.json --require-ocr

Read-only and stdlib-only. Checks local files, not CUDA, model weights or GPU RAM.
"""
import argparse
import json
from pathlib import Path

TASK = "key information extraction en"

def check(rows, root, require_ocr):
    if not isinstance(rows, list) or len(rows) != 400:
        raise ValueError(f"Expected exactly 400 rows; found {len(rows) if isinstance(rows, list) else type(rows).__name__}")
    seen = set()
    text_paths = set()
    problems = []
    for i, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            problems.append(f"row {i}: not a JSON object")
            continue
        needed = ("id", "type", "dataset_name", "image_path", "question", "answers")
        missing = [k for k in needed if k not in row or row[k] is None]
        if missing:
            problems.append(f"row {i}: missing {missing}")
            continue
        if row["type"] != TASK:
            problems.append(f"row {i}: unexpected type {row['type']!r}")
        key = (str(row["dataset_name"]), str(row["id"]), str(row["image_path"]))
        if key in seen:
            problems.append(f"row {i}: duplicate record {key}")
        seen.add(key)
        image = Path(row["image_path"])
        if image.is_absolute() or ".." in image.parts:
            problems.append(f"row {i}: unsafe image path {image}")
        elif not ((root / image).is_file() or (root.parent / image).is_file()):
            problems.append(f"row {i}: image missing {image}")
        if require_ocr:
            value = row.get("text_path")
            if not isinstance(value, str) or not value:
                problems.append(f"row {i}: missing text_path")
                continue
            txt = Path(value)
            if txt.is_absolute() or ".." in txt.parts or txt.suffix != ".txt":
                problems.append(f"row {i}: unsafe text_path {value!r}")
                continue
            if str(txt) in text_paths:
                problems.append(f"row {i}: duplicate OCR path {value}")
            text_paths.add(str(txt))
            if not txt.is_file():
                problems.append(f"row {i}: OCR text missing {value}")
    return problems

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=Path("data/OCRBench_v2"))
    p.add_argument("--require-ocr", action="store_true")
    args = p.parse_args()
    if not args.input.is_file():
        p.error(f"Benchmark manifest not found: {args.input}")
    with args.input.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    issues = check(rows, args.dataset_root, args.require_ocr)
    if issues:
        for issue in issues[:20]:
            print("FAIL:", issue)
        raise SystemExit(f"{len(issues)} issue(s) found. Correct these before GPU inference.")
    print(f"PASS: 400 unique English KIE records, 400 images, OCR text required={args.require_ocr}")
if __name__ == "__main__":
    main()
