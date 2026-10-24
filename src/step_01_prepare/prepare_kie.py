"""Select a single OCRBench v2 English KIE benchmark for every model.

This script reads the original annotations and writes a new JSON file. It does
not change the original data, copy images, run OCR, or evaluate predictions.

Example (from the repository root):
    python src/step_01_prepare/prepare_kie.py \
      --input data/OCR_Bench_v2/OCRBench_v2.json \
      --output data/benchmark/english_kie_400.json \
      --expected-count 400
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random

TASK_TYPE = "key information extraction en"
REQUIRED_FIELDS = ("id", "type", "dataset_name", "image_path", "question", "answers")


def select_kie_examples(records, dataset_name=None, limit=None, seed=42, ocr_data=None):
    """Return matching original records without changing their contents."""
    if not isinstance(records, list):
        raise ValueError("OCRBench annotations must be a JSON list")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")

    selected = [
        record for record in records
        if isinstance(record, dict)
        and record.get("type") == TASK_TYPE
        and (dataset_name is None or str(record.get("dataset_name", "")).casefold() == dataset_name.casefold())
    ]

    if limit is not None and len(selected) > limit:
        selected = random.Random(seed).sample(selected, limit)

    # if ocr_data argument is not null, add text_path to each record
    if ocr_data:
        selected = [
            {
                **record,
                "text_path": str(
                    Path(ocr_data) / f"{Path(record['image_path']).stem}.txt"
                ),
            }
            for record in selected
        ]  
        
    return selected


def validate_selection(records, expected_count=None):
    """Check that selected records are identifiable, distinct and complete."""
    if not records:
        raise ValueError("No matching English KIE records found")
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} records, found {len(records)}")

    identities = set()
    for position, record in enumerate(records, start=1):
        missing = [name for name in REQUIRED_FIELDS if name not in record or record[name] is None]
        if missing:
            raise ValueError(f"Record {position}: missing required fields {missing}")
        if not isinstance(record["image_path"], str) or not record["image_path"].strip():
            raise ValueError(f"Record {position}: invalid image_path")
        if not isinstance(record["question"], str):
            raise ValueError(f"Record {position}: question must be text")
        if record["type"] != TASK_TYPE:
            raise ValueError(f"Record {position}: wrong task type")

        identity = (str(record["dataset_name"]), str(record["id"]), record["image_path"])
        if identity in identities:
            raise ValueError(f"Duplicate document identity: {identity}")
        identities.add(identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original OCRBench_v2.json")
    parser.add_argument("--output", type=Path, required=True, help="Prepared English KIE JSON")
    parser.add_argument("--ocr_data", type=Path, required=False, help="where ocr_output is located")
    parser.add_argument("--dataset-name", help="Optional source dataset, e.g. POIE")
    parser.add_argument("--limit", type=int, help="Optional number of examples for a pilot")
    parser.add_argument("--seed", type=int, default=42, help="Seed for pilot sampling")
    parser.add_argument("--expected-count", type=int, help="Require exactly this many selected records")
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        parser.error("--output must not overwrite the original annotations")
    if args.expected_count is not None and args.expected_count < 1:
        parser.error("--expected-count must be positive")

    with args.input.open(encoding="utf-8") as file:
        original = json.load(file)
    selected = select_kie_examples(original, args.dataset_name, args.limit, args.seed, args.ocr_data)
    validate_selection(selected, args.expected_count)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(selected, file, indent=2, ensure_ascii=False)
        file.write("\n")

    counts = Counter(str(row["dataset_name"]) for row in selected)
    digest = sha256(args.output.read_bytes()).hexdigest()
    print(f"Selected {len(selected)} English KIE documents -> {args.output}")
    print("By dataset:", dict(sorted(counts.items())))
    print("SHA256:", digest)
    print("Use this same prepared JSON for every model and OCR pipeline.")


if __name__ == "__main__":
    main()
