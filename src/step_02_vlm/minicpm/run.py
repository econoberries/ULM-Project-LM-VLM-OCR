"""
Run MiniCPM-V-4.6 on all English KIE records in an OCRBench v2 JSON file.
uv run python src/step_02_vlm/minicpm/run.py

References
1. OCRB, OCRBench inference example: https://github.com/Yuliang-Liu/MultimodalOCR/blob/4e363b3f499811b43d792ef0bb33932ef05eca27/OCRBench/example.py
2. OCRB2, OCRBench v2 evaluator: https://github.com/Yuliang-Liu/MultimodalOCR/blob/4e363b3f499811b43d792ef0bb33932ef05eca27/OCRBench_v2/eval_scripts/eval.py
"""

import copy
import logging
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

from minicpm_utils import (
    DEFAULT_OUTPUT_FOLDER,
    MODEL_ID,
    MODEL_REVISION,
    environment_info,
    load_model,
    run_inference,
    run_metadata,
    start_sample_timer,
    stop_sample_timer,
)
from ocrbench_utils import (
    DEFAULT_IMAGE_FOLDER,
    DEFAULT_INPUT_FILE,
    KIE_TYPE,
    check_not_same_file,
    load_done_predictions,
    record_key,
    load_json,
    resolve_image_path,
    save_json,
    sidecar_path,
    validate_predictions,
)

logger = logging.getLogger("run")


# OCRB, lines 28-37
def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, default=str(DEFAULT_INPUT_FILE))
    parser.add_argument("--image_folder", type=str, default=str(DEFAULT_IMAGE_FOLDER))
    parser.add_argument("--output_file", type=str, default=None,
                        help="Default: results/minicpm/<input name>_minicpm_v46.json")
    parser.add_argument("--model_path", type=str, default=MODEL_ID)
    parser.add_argument("--revision", type=str, default=MODEL_REVISION)
    args = parser.parse_args()
    if args.output_file is None:
        args.output_file = str(DEFAULT_OUTPUT_FOLDER / f"{Path(args.input_file).stem}_minicpm_v46.json")
    return args


def setup_logging(log_file):
    # Console + <output>.log (git-ignored).
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
    )


def main():
    args = _get_args()
    setup_logging(Path(args.output_file).with_suffix(".log"))
    check_not_same_file(args.input_file, args.output_file)

    data = load_json(args.input_file)
    kie_data = [item for item in data if item["type"] == KIE_TYPE]
    logger.info(f"{len(kie_data)} of {len(data)} rows in {args.input_file} have type {KIE_TYPE!r}")

    # OCRB, lines 92-96
    # Resume, matched by the full benchmark record identity.
    done = load_done_predictions(args.output_file)
    if done:
        logger.info(f"{args.output_file} exists: {len(done)} rows already have a prediction and are skipped")

    errors_file = sidecar_path(args.output_file, "errors")
    errors = []
    model = None
    if any(record_key(item) not in done for item in kie_data):
        # Load once, outside the loop.
        load_start = time.perf_counter()
        model, processor = load_model(args.model_path, args.revision)
        load_seconds = round(time.perf_counter() - load_start, 3)

        results = []
        for i, record in enumerate(kie_data):
            prefix = f"[{i + 1}/{len(kie_data)}] id={record['id']} dataset={record['dataset_name']}"
            if record_key(record) in done:
                logger.info(f"{prefix} skipped (already predicted)")
                results.append(done[record_key(record)])
                continue

            try:
                # OCRB, lines 59-60
                img_path = resolve_image_path(args.image_folder, record["image_path"])
                start = start_sample_timer()
                response, n_tokens = run_inference(model, processor, img_path, record["question"])
                seconds, peak = stop_sample_timer(start)
            except Exception as e:
                # OCRB2, lines 236-245
                # Failed records stay out of the output; details go to the errors file.
                logger.exception(f"{prefix} FAILED")
                errors.append({"id": record["id"], "image_path": record["image_path"],
                               "error": f"{type(e).__name__}: {e}"})
                save_json(errors, errors_file)
                continue

            # OCRB, line 84
            # Copy, so the input data stays unchanged.
            output_record = copy.deepcopy(record)
            output_record["predict"] = response
            output_record.update(inference_seconds=seconds, peak_allocated_gpu_bytes=peak, output_tokens=n_tokens)
            logger.info(f"{prefix} ok, {seconds} s, {n_tokens} tokens, response starts with: {response[:60]!r}")
            results.append(output_record)
            # Save after every record.
            save_json(results, args.output_file)
    else:
        logger.info("Nothing left to predict.")

    if not errors:
        # Stale errors file from an earlier run.
        errors_file.unlink(missing_ok=True)

    # Re-read the output; check every KIE record.
    failed_ids = validate_predictions(kie_data, args.output_file)

    # Metadata only for a run that loaded the model.
    if model is not None:
        save_json(run_metadata(model, args.model_path, args.revision, input_file=args.input_file,
                               image_folder=args.image_folder, num_examples=len(kie_data),
                               num_predicted=len(kie_data) - len(failed_ids), failed_ids=failed_ids,
                               output_file=args.output_file, model_load_seconds=load_seconds,
                               **environment_info()),
                  sidecar_path(args.output_file, "run_metadata"))

    return 1 if failed_ids else 0


if __name__ == "__main__":
    sys.exit(main())
