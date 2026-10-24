"""Summarize Qwen prediction logs and optionally official OCRBench scores."""

import argparse
import json
from pathlib import Path
from statistics import mean, median


def summarize(rows):
    failures = sum(bool(row.get("error")) for row in rows)
    durations = [row["inference_seconds"] for row in rows
                 if isinstance(row.get("inference_seconds"), (int, float))]
    scores = [row["score"] for row in rows
              if isinstance(row.get("score"), (int, float))]
    return {
        "samples": len(rows),
        "failures": failures,
        "timed_samples": len(durations),
        "mean_inference_seconds": mean(durations) if durations else None,
        "median_inference_seconds": median(durations) if durations else None,
        "official_scored_samples": len(scores),
        "mean_official_kie_f1": mean(scores) if scores else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError("Expected OCRBench result list")
    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()
