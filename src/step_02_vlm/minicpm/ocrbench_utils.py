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
DEFAULT_INPUT_FILE = REPO_ROOT / "data" / "smoke_test" / "kie_en_smoke10.json"
# image_path in the JSON is relative to this folder
DEFAULT_IMAGE_FOLDER = REPO_ROOT / "data" / "smoke_test"

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
    full_path = os.path.join(image_folder, image_path)
    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"Image not found: {full_path}")
    return full_path


# OCRB2, lines 241-245
def has_valid_predict(record):
    return isinstance(record.get("predict"), str)


# OCRB, lines 92-96
def load_done_predictions(output_file):
    """{id: record} for output records that already have a valid `predict`."""
    if not os.path.exists(output_file):
        return {}
    return {item["id"]: item for item in load_json(output_file) if has_valid_predict(item)}


def validate_predictions(expected_records, output_file):
    """Return ids without a string `predict` in the output file (empty list = OK)."""
    done = load_done_predictions(output_file)
    failed_ids = [item["id"] for item in expected_records if item["id"] not in done]
    if failed_ids:
        logger.error(f"VALIDATION FAILED: {len(failed_ids)}/{len(expected_records)} rows have no string "
                     f"`predict`: ids={failed_ids}. See {sidecar_path(output_file, 'errors')}.")
        logger.error(f"Do NOT run the official OCRBench evaluation on {output_file}; "
                     f"rerun the same command to retry the failed rows first.")
    else:
        logger.info(f"VALIDATION OK: all {len(expected_records)} rows have a string `predict` in {output_file}")
    return failed_ids
