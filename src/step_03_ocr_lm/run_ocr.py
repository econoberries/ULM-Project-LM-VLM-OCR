"""Run PaddleOCR on exactly the documents in our shared OCR manifest.

From the repository root, on a suitable compute node:

    python src/step_03_ocr_lm/run_ocr.py \
      --input data/benchmark/english_kie_400_ocr.json \
      --dataset-root data/OCRBench_v2

For a small test, add: --limit 1

Requires PaddleOCR and a compatible PaddlePaddle installation. This is the
OCR stage ONLY: it does not call the language models or score their answers.
The manifest defines one unique text_path per document. We write .txt and a
neighboring .json containing recognized text, boxes and confidence scores.
"""

import argparse
import json
import os
from pathlib import Path
import tempfile
import time


TASK_TYPE = "key information extraction en"
DEFAULT_REPORT = Path("data/ocr_output/paddleocr_metrics.json")


def save_json(path, data):
    """Atomically save JSON so an interrupted job cannot corrupt the report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as file:
        temp_path = Path(file.name)
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temp_path, path)


def load_records(path):
    with path.open(encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list) or not records:
        raise ValueError("OCR input manifest must be a nonempty JSON list")
    seen = set()
    for number, row in enumerate(records, start=1):
        if not isinstance(row, dict) or row.get("type") != TASK_TYPE:
            raise ValueError(f"Record {number} is not an English KIE document")
        if not all(row.get(key) is not None for key in ("id", "dataset_name", "image_path", "text_path")):
            raise ValueError(f"Record {number} lacks a required field")
        if not isinstance(row["image_path"], str) or not row["image_path"].strip():
            raise ValueError(f"Record {number} has invalid image_path")
        out = Path(row["text_path"])
        if (out.is_absolute() or ".." in out.parts or out.parts[:2] != ("data", "ocr_output")
                or out.suffix.lower() != ".txt"):
            raise ValueError(f"Record {number} has unsafe or unexpected text_path: {out}")
        if out.as_posix() in seen:
            raise ValueError(f"Two documents share OCR output path: {out}")
        seen.add(out.as_posix())
    return records


def image_file(dataset_root, image_path):
    """OCRBench may specify paths relative to either the dataset or its parent."""
    raw = Path(image_path)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"Unsafe image path: {image_path}")
    for candidate in (dataset_root / raw, dataset_root.parent / raw):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find image {image_path!r} under {dataset_root}")


def extract_lines(prediction):
    """Read PaddleOCR's recognized text and coordinates (PaddleOCR 3.x)."""
    lines = []
    for result in prediction:
        data = getattr(result, "json", None)
        if callable(data):
            data = data()
        if not isinstance(data, dict):
            raise TypeError("Unexpected PaddleOCR output; expected result.json dictionary")
        data = data.get("res", data)
        texts = data.get("rec_texts", [])
        boxes = data.get("rec_boxes")
        scores = data.get("rec_scores", [])
        polygons = data.get("rec_polys")
        if hasattr(boxes, "tolist"):
            boxes = boxes.tolist()
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if hasattr(polygons, "tolist"):
            polygons = polygons.tolist()
        for index, text in enumerate(texts):
            line = {"text": str(text), "confidence": float(scores[index]) if index < len(scores) else None}
            if boxes is not None and index < len(boxes):
                line["bbox"] = [float(x) for x in boxes[index][:4]]
            elif polygons is not None and index < len(polygons):
                pts = polygons[index]
                xs, ys = [float(p[0]) for p in pts], [float(p[1]) for p in pts]
                line["bbox"] = [min(xs), min(ys), max(xs), max(ys)]
            else:
                line["bbox"] = None
            lines.append(line)
    return lines


def format_text(lines):
    """Group detected lines by vertical position, then sort left-to-right."""
    if not lines:
        return ""
    positioned = [x for x in lines if x["bbox"] and len(x["bbox"]) == 4]
    if not positioned:
        return "\n".join(x["text"] for x in lines)
    heights = [max(1, x["bbox"][3] - x["bbox"][1]) for x in positioned]
    tolerance = max(1, sum(heights) / len(heights) * 0.5)
    rows = []
    for line in sorted(positioned, key=lambda x: (x["bbox"][1] + x["bbox"][3]) / 2):
        y = (line["bbox"][1] + line["bbox"][3]) / 2
        group = next((row for row in rows if abs(row["y"] - y) <= tolerance), None)
        if group is None:
            rows.append({"y": y, "lines": [line]})
        else:
            group["lines"].append(line)
            group["y"] = sum((x["bbox"][1] + x["bbox"][3]) / 2 for x in group["lines"]) / len(group["lines"])
    output = [" ".join(item["text"] for item in sorted(row["lines"], key=lambda x: x["bbox"][0]))
              for row in rows]
    output.extend(line["text"] for line in lines if line["bbox"] is None)
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Output from prepare_ocr_manifest.py")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Original OCRBench_v2 directory")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Per-document OCR execution metrics")
    parser.add_argument("--limit", type=int, help="Process only the first N documents as a test")
    parser.add_argument("--force", action="store_true", help="Reprocess existing OCR text and JSON files")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    records = load_records(args.input)
    selected = records[:args.limit] if args.limit else records
    if args.report.resolve() == args.input.resolve():
        parser.error("--report must not overwrite the manifest")
    if not selected:
        raise ValueError("No records to process")

    # Validate selected image paths before loading the expensive OCR model.
    images = [image_file(args.dataset_root, row["image_path"]) for row in selected]
    pending = [i for i, row in enumerate(selected) if args.force or
               not (Path(row["text_path"]).is_file() and Path(row["text_path"]).with_suffix(".json").is_file())]
    if not pending:
        print(f"All {len(selected)} documents already have OCR text and JSON files; nothing to process")
        return

    # Import here so validation can run without installing PaddleOCR locally.
    from paddleocr import PaddleOCR
    load_start = time.perf_counter()
    # PaddleOCR 3.x supports device="gpu:0"; require GPU inference rather
    # than silently falling back to CPU on the allocated A100/H100 node.
    ocr = PaddleOCR(lang="en", device="gpu:0", use_doc_orientation_classify=False,
                    use_doc_unwarping=False, use_textline_orientation=False)
    load_seconds = time.perf_counter() - load_start
    print(f"PaddleOCR loaded in {load_seconds:.2f}s; processing {len(pending)} documents", flush=True)

    previous = {}
    if args.report.exists():
        with args.report.open(encoding="utf-8") as file:
            old = json.load(file)
        previous = {row["text_path"]: row for row in old.get("records", [])}
    metrics = dict(previous)
    for i in pending:
        row = selected[i]
        target = Path(row["text_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        info = {"id": row["id"], "dataset_name": row["dataset_name"],
                "image_path": row["image_path"], "text_path": target.as_posix()}
        try:
            start = time.perf_counter()
            prediction = ocr.predict(str(images[i]))
            inference_seconds = time.perf_counter() - start
            lines = extract_lines(prediction)
            formatted = format_text(lines)
            start_write = time.perf_counter()
            # The companion JSON keeps layout coordinates for future experiments.
            save_json(target.with_suffix(".json"), {"image_path": row["image_path"],
                                                    "lines": lines, "formatted_text": formatted})
            target.write_text(formatted, encoding="utf-8")
            write_seconds = time.perf_counter() - start_write
            info.update(status="ok", ocr_inference_seconds=round(inference_seconds, 4),
                        write_seconds=round(write_seconds, 4), characters=len(formatted),
                        detected_lines=len(lines), error=None)
            print(f"[{i+1}/{len(selected)}] OK {row['dataset_name']} {row['id']}", flush=True)
        except Exception as error:
            info.update(status="error", error=f"{type(error).__name__}: {error}")
            print(f"[{i+1}/{len(selected)}] ERROR {row['dataset_name']} {row['id']}: {error}", flush=True)
        metrics[target.as_posix()] = info
        save_json(args.report, {"ocr_engine": "PaddleOCR", "device": "gpu:0", "model_load_seconds": round(load_seconds, 4),
                                "records": list(metrics.values())})

    failures = [x for x in selected if metrics.get(x["text_path"], {}).get("status") != "ok"]
    print(f"OCR stage: {len(selected) - len(failures)}/{len(selected)} completed successfully")
    print(f"Per-document measurements: {args.report}")
    if failures:
        raise SystemExit(f"{len(failures)} OCR documents failed. Inspect the report and rerun.")


if __name__ == "__main__":
    main()
