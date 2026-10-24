"""Read-only checks before submitting the full Qwen3.5 OCRBench KIE job.

From the repository root (after preparing the 400-record manifest):
    uv run --locked python tests/check_qwen_setup.py --check-deps

This script does not load Qwen, reserve a GPU, download data or modify files.
A successful check does NOT prove that model weights or CUDA will work on Helix.
"""

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

TASK = "key information extraction en"
REQUIRED_FILES = (
    "src/step_02_vlm/qwen35/run.py",
    "slurm/vlm/qwen35_kie.sbatch",
    "pyproject.toml",
    "uv.lock",
)


def resolve_image(dataset_root, image_path):
    """Match the original Qwen runner's image path resolution."""
    raw = Path(image_path)
    candidates = [raw] if raw.is_absolute() else [dataset_root / raw, dataset_root.parent / raw]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def validate_manifest(records, dataset_root, expected_count):
    """Return (errors, counts) without changing benchmark records."""
    errors = []
    counts = Counter()
    if not isinstance(records, list):
        return ["Prepared input JSON must be a list"], counts
    if len(records) != expected_count:
        errors.append(f"Expected {expected_count} documents; found {len(records)}")
    seen = set()
    missing_images = []
    for position, row in enumerate(records, start=1):
        if not isinstance(row, dict):
            errors.append(f"Record {position} is not a JSON object")
            continue
        required = ("id", "type", "dataset_name", "image_path", "question", "answers")
        absent = [key for key in required if row.get(key) is None]
        if absent:
            errors.append(f"Record {position} missing: {', '.join(absent)}")
            continue
        if row["type"] != TASK:
            errors.append(f"Record {position} is not English KIE")
        if not isinstance(row["question"], str):
            errors.append(f"Record {position} question must be text")
        if not isinstance(row["image_path"], str) or not row["image_path"].strip():
            errors.append(f"Record {position} image_path is invalid")
            continue
        # Identical to the composite identity used by the original Qwen runner.
        identity = json.dumps([row.get("dataset_name"), row.get("type"), row.get("id"),
                               row.get("image_path"), row.get("question")], ensure_ascii=False)
        if identity in seen:
            errors.append(f"Duplicate document identity at record {position}")
        seen.add(identity)
        counts[str(row["dataset_name"])] += 1
        if resolve_image(dataset_root, row["image_path"]) is None:
            missing_images.append((position, row["image_path"]))
    if missing_images:
        preview = "; ".join(f"#{i}: {path}" for i, path in missing_images[:5])
        errors.append(f"Missing {len(missing_images)} image files (first five: {preview})")
    return errors, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input", type=Path, default=Path("data/benchmark/english_kie_400.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/OCRBench_v2"))
    parser.add_argument("--expected-count", type=int, default=400)
    parser.add_argument("--check-deps", action="store_true", help="Check installed Python package availability")
    args = parser.parse_args()
    if args.expected_count < 1:
        parser.error("--expected-count must be positive")

    root = args.repo_root.resolve()
    input_file = args.input if args.input.is_absolute() else root / args.input
    dataset_root = args.dataset_root if args.dataset_root.is_absolute() else root / args.dataset_root
    errors = []

    print("Qwen3.5 preflight (no GPU inference)")
    print("Repository:", root)
    for relative in REQUIRED_FILES:
        if not (root / relative).is_file():
            errors.append(f"Missing repository file: {relative}")
    if sys.version_info[:2] != (3, 12):
        errors.append(f"Expected Python 3.12 per pyproject.toml; found {sys.version.split()[0]}")
    if args.check_deps:
        for name in ("torch", "transformers", "accelerate", "PIL"):
            if importlib.util.find_spec(name) is None:
                errors.append(f"Python package not installed: {name}")

    if not input_file.is_file():
        errors.append(f"Prepared benchmark missing: {input_file}")
    else:
        try:
            with input_file.open(encoding="utf-8") as stream:
                rows = json.load(stream)
            manifest_errors, counts = validate_manifest(rows, dataset_root, args.expected_count)
            errors.extend(manifest_errors)
            print("Datasets:", dict(sorted(counts.items())))
        except (OSError, ValueError, TypeError) as error:
            errors.append(f"Unable to inspect manifest: {error}")

    if errors:
        print(f"FAIL — {len(errors)} issue(s):")
        for item in errors:
            print(" -", item)
        return 1
    print("PASS — prepared documents, image paths and requested local checks look valid.")
    print("Still required on a GPU node: confirm model load, CUDA compatibility and generation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
