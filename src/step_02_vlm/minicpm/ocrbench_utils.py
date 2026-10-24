"""
OCRBench v2  helpers for the MiniCPM: JSON I/O, image paths, resume, validation.

Per run:
<name>.json - OCRBench predictions
<name>.errors.json - failed records
<name>.run_metadata.json - run settings and environment

References
1. OCRB, OCRBench inference example: https://github.com/Yuliang-Liu/MultimodalOCR/blob/4e363b3f499811b43d792ef0bb33932ef05eca27/OCRBench/example.py
2. OCRB2, OCRBench v2 evaluator: https://github.com/Yuliang-Liu/MultimodalOCR/blob/4e363b3f499811b43d792ef0bb33932ef05eca27/OCRBench_v2/eval_scripts/eval.py
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# src/vlm/minicpm/ -> repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_FILE = REPO_ROOT / "data" / "benchmark" / "english_kie_400.json"
# image_path in the JSON is relative to this folder
DEFAULT_IMAGE_FOLDER = REPO_ROOT / "data" / "OCR_Bench_v2"

# OCRB2, lines 39, 236
KIE_TYPE = "key information extraction en"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# OCRB, lines 24-26
def save_json(obj, save_path):
    """Save as UTF-8 via a temp file, so a killed job never leaves it half-written."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = save_path.with_name(save_path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)
    os.replace(tmp_path, save_path)


def sidecar_path(output_file, kind):
    """x.json -> x.<kind>.json (kind: "errors" or "run_metadata")."""
    return Path(output_file).with_suffix(f".{kind}.json")


def check_not_same_file(input_file, output_file):
    """Refuse to overwrite the input file."""
    if Path(input_file).resolve() == Path(output_file).resolve():
        raise ValueError(f"Output file must differ from input file: {input_file}")


# OCRB, line 59
def resolve_image_path(image_folder, image_path):
    raw = Path(image_path)
    root = Path(image_folder)
    candidates = [raw] if raw.is_absolute() else [root / raw, root.parent / raw]
    if not raw.is_absolute() and raw.parts and raw.parts[0] == "OCRBench_v2":
        candidates.append(root.joinpath(*raw.parts[1:]))
    for path in candidates:
        if path.is_file():
            return str(path)
    raise FileNotFoundError(f"Image not found: {image_path!r}; tried {candidates}")


# OCRB2, lines 241-245
def has_valid_predict(record):
    return isinstance(record.get("predict"), str)


# OCRB, lines 92-96
def record_key(record):
    """Composite identity: dataset, task, id, image path and question."""
    return json.dumps(
        [record.get("dataset_name"), record.get("type"), record.get("id"),
         record.get("image_path"), record.get("question")],
        ensure_ascii=False,
    )


def load_done_predictions(output_file):
    """Resume only exact benchmark records with a string prediction."""
    if not os.path.exists(output_file):
        return {}
    done = {}
    for item in load_json(output_file):
        if not has_valid_predict(item):
            continue
        key = record_key(item)
        if key in done:
            raise ValueError(f"Duplicate prediction in {output_file}: {key}")
        done[key] = item
    return done


def validate_predictions(expected_records, output_file):
    """Return ids of missing exact records (empty list = OK)."""
    done = load_done_predictions(output_file)
    failed_ids = [item["id"] for item in expected_records if record_key(item) not in done]
    if failed_ids:
        logger.error(f"VALIDATION FAILED: {len(failed_ids)}/{len(expected_records)} rows have no string "
                     f"`predict`: ids={failed_ids}. See {sidecar_path(output_file, 'errors')}.")
        logger.error(f"Do NOT run the official OCRBench evaluation on {output_file}; "
                     f"rerun the same command to retry the failed rows first.")
    else:
        logger.info(f"VALIDATION OK: all {len(expected_records)} rows have a string `predict` in {output_file}")
    return failed_ids
