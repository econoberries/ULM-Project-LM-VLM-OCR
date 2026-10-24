"""Select reproducible OCRBench v2 English KIE examples without editing source data."""

import argparse
import json
import random
from pathlib import Path

ENGLISH_KIE_TYPE = "key information extraction en"


def select_kie_examples(records, dataset_name=None, limit=None, seed=42):
    """Select English KIE entries and preserve the original benchmark fields."""
    if not isinstance(records, list):
        raise ValueError("OCRBench annotations must be a JSON list of records")
    matches = [
        item for item in records
        if isinstance(item, dict)
        and item.get("type") == ENGLISH_KIE_TYPE
        and (dataset_name is None or str(item.get("dataset_name", "")).casefold() == dataset_name.casefold())
    ]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1 (omit it for all examples)")
        if len(matches) > limit:
            matches = random.Random(seed).sample(matches, limit)
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="OCRBench_v2.json")
    parser.add_argument("--output", type=Path, required=True, help="Selected examples JSON")
    parser.add_argument("--dataset-name", help="Optional benchmark dataset_name, e.g. POIE")
    parser.add_argument("--limit", type=int, help="Maximum examples (omit for all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as handle:
        records = json.load(handle)
    selected = select_kie_examples(records, args.dataset_name, args.limit, args.seed)
    if not selected:
        raise SystemExit("No English KIE records matched. Inspect the input JSON and dataset_name.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(selected, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Selected {len(selected)} / {len(records)} records -> {args.output}")
    print("Datasets:", sorted({str(item.get('dataset_name', '?')) for item in selected}))


if __name__ == "__main__":
    main()
