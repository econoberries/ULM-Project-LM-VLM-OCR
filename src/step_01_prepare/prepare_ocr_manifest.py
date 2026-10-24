"""Add collision-resistant OCR text paths to our shared English KIE selection.

Run from the repository root *after* prepare_kie.py:

    python src/step_01_prepare/prepare_ocr_manifest.py \
      --input data/benchmark/english_kie_400.json \
      --output data/benchmark/english_kie_400_ocr.json \
      --expected-count 400

This prepares an OCR input manifest. It does NOT read images, run OCR, create
text files, or alter the original benchmark records. The resulting JSON still
contains benchmark annotations, so keep it locally (ignored by Git).

Later, the OCR program should read each row's image_path and save the detected
text at exactly that row's text_path.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

TASK_TYPE = "key information extraction en"
REQUIRED_FIELDS = ("id", "dataset_name", "type", "image_path", "question", "answers")


def safe_part(value):
    """Make a readable filesystem-safe folder or filename component."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_")
    if not cleaned:
        raise ValueError(f"Invalid filename component: {value!r}")
    return cleaned


def build_ocr_manifest(records, ocr_dir="data/ocr_output"):
    """Return copies of selected records with unique, deterministic text paths."""
    if not isinstance(records, list) or not records:
        raise ValueError("Input must be a nonempty JSON list")

    result = []
    seen_documents = set()
    seen_paths = set()

    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Record {position} is not a JSON object")
        missing = [field for field in REQUIRED_FIELDS if field not in record or record[field] is None]
        if missing:
            raise ValueError(f"Record {position} has missing fields: {missing}")
        if record["type"] != TASK_TYPE:
            raise ValueError(f"Record {position} is not English KIE")
        if not isinstance(record["image_path"], str) or not record["image_path"].strip():
            raise ValueError(f"Record {position} has an invalid image_path")

        dataset = str(record["dataset_name"])
        image_path = record["image_path"]
        identity = (dataset, str(record["id"]), image_path)
        if identity in seen_documents:
            raise ValueError(f"Duplicate benchmark document: {identity}")
        seen_documents.add(identity)

        # Include the source image path to avoid collisions between images with
        # identical basenames in different OCRBench folders.
        short_hash = sha256(json.dumps(identity, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
        text_path = Path(ocr_dir) / safe_part(dataset.lower()) / f"{safe_part(record['id'])}_{short_hash}.txt"
        if text_path.as_posix() in seen_paths:
            raise ValueError(f"OCR text path collision: {text_path}")
        seen_paths.add(text_path.as_posix())

        prepared = record.copy()
        prepared["text_path"] = text_path.as_posix()
        result.append(prepared)

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Shared prepared English KIE JSON")
    parser.add_argument("--output", required=True, type=Path, help="OCR-enriched English KIE JSON")
    parser.add_argument("--ocr-dir", default="data/ocr_output", help="Local directory for future OCR text files")
    parser.add_argument("--expected-count", type=int, default=None, help="Expected number of examples")
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        parser.error("--output must not overwrite --input")
    if args.expected_count is not None and args.expected_count < 1:
        parser.error("--expected-count must be positive")
    if Path(args.ocr_dir).is_absolute() or ".." in Path(args.ocr_dir).parts:
        parser.error("--ocr-dir must be a relative path inside the repository")

    with args.input.open(encoding="utf-8") as file:
        selected = json.load(file)
    manifest = build_ocr_manifest(selected, args.ocr_dir)
    if args.expected_count is not None and len(manifest) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} examples, found {len(manifest)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(f"Prepared OCR paths for {len(manifest)} documents -> {args.output}")
    print("Example image:", manifest[0]["image_path"])
    print("Example OCR text destination:", manifest[0]["text_path"])
    print("No OCR has been run; these are destination paths only.")


if __name__ == "__main__":
    main()
