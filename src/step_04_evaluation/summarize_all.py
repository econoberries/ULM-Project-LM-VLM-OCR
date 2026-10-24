"""
Summarize a prediction file of any pipeline (MiniCPM, Qwen3.5, OCR + LM), including official KIE F1, time, GPU memory
"""

import argparse
import json
from pathlib import Path
from statistics import mean, median


def summarize(rows, metadata=None):
    metadata = metadata or {}
    # failures also count MiniCPM's failed_ids from metadata
    failures = sum(bool(row.get("error")) for row in rows) + len(metadata.get("failed_ids", []))
    durations = [row["inference_seconds"] for row in rows
                 if isinstance(row.get("inference_seconds"), (int, float))]
    scores = [row["score"] for row in rows
              if isinstance(row.get("score"), (int, float))]
    # Per-record peak torch.cuda.max_memory_allocated, one value per GPU
    peaks_gib = [max(row["peak_allocated_gpu_bytes"]) / 2**30 for row in rows
                 if row.get("peak_allocated_gpu_bytes")]
    return {
        "samples": len(rows),
        "failures": failures,
        "timed_samples": len(durations),
        "mean_inference_seconds": mean(durations) if durations else None,
        "median_inference_seconds": median(durations) if durations else None,
        "official_scored_samples": len(scores),
        "mean_official_kie_f1": mean(scores) if scores else None,
        # newly added
        "mean_peak_allocated_gpu_gib": mean(peaks_gib) if peaks_gib else None,
        "max_peak_allocated_gpu_gib": max(peaks_gib) if peaks_gib else None,
        "gpu_names": metadata.get("cuda_device_names"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError("Expected OCRBench result list")
    metadata = None
    if args.metadata:
        with args.metadata.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
    summary = summarize(rows, metadata)
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
