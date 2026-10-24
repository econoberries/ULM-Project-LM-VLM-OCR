"""
Run MiniCPM-V-4.6 on a single OCRBench v2 KIE example and save the prediction
uv run python src/vlm/minicpm/run_one_record.py

References
1. OCRB, OCRBench inference example: https://github.com/Yuliang-Liu/MultimodalOCR/blob/4e363b3f499811b43d792ef0bb33932ef05eca27/OCRBench/example.py
2. OCRB2, OCRBench v2 evaluator: https://github.com/Yuliang-Liu/MultimodalOCR/blob/4e363b3f499811b43d792ef0bb33932ef05eca27/OCRBench_v2/eval_scripts/eval.py
"""

import copy
import logging
import time
from argparse import ArgumentParser

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
    load_json,
    resolve_image_path,
    save_json,
    sidecar_path,
)


# OCRB, lines 28-37
def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, default=str(DEFAULT_INPUT_FILE))
    parser.add_argument("--image_folder", type=str, default=str(DEFAULT_IMAGE_FOLDER))
    parser.add_argument("--output_file", type=str, default=None,
                        help="Default: results/minicpm/single_id<sample_id>.json")
    parser.add_argument("--sample_id", type=int, default=900)
    parser.add_argument("--model_path", type=str, default=MODEL_ID)
    parser.add_argument("--revision", type=str, default=MODEL_REVISION)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _get_args()
    output_file = args.output_file or str(DEFAULT_OUTPUT_FOLDER / f"single_id{args.sample_id}.json")
    check_not_same_file(args.input_file, output_file)

    # 1) Find the record
    data = load_json(args.input_file)
    matches = [item for item in data if item["id"] == args.sample_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one record with id={args.sample_id}, found {len(matches)}")
    record = matches[0]

    # 2) KIE extraction records only
    if record["type"] != KIE_TYPE:
        raise ValueError(f"Record {args.sample_id} has type {record['type']!r}, expected {KIE_TYPE!r}")

    # 3) OCRB, lines 59-60
    # Image path joined
    img_path = resolve_image_path(args.image_folder, record["image_path"])
    qs = record["question"]
    print(f"id={record['id']}  dataset={record['dataset_name']}  image={img_path}")
    print(f"----- question -----\n{qs}")

    # 4) Load the model, then time the inference.
    load_start = time.perf_counter()
    model, processor = load_model(args.model_path, args.revision)
    load_seconds = round(time.perf_counter() - load_start, 3)

    start = start_sample_timer()
    response, n_tokens = run_inference(model, processor, img_path, qs)
    seconds, peak = stop_sample_timer(start)

    print(f"----- raw model response -----\n{response}")
    print(f"----- {seconds} s, {n_tokens} output tokens, peak GPU bytes {peak} -----")
    print(f"----- ground truth (answers) -----\n{record['answers']}")

    # 5) OCRB, line 84 &
    # OCRB2, lines 34-48
    # Copy with predict
    output_record = copy.deepcopy(record)
    output_record.update(predict=response, inference_seconds=seconds,
                         peak_allocated_gpu_bytes=peak, output_tokens=n_tokens)
    save_json([output_record], output_file)
    save_json(run_metadata(model, args.model_path, args.revision, input_file=args.input_file,
                           image_folder=args.image_folder, num_examples=1, sample_id=args.sample_id,
                           output_file=output_file, model_load_seconds=load_seconds, **environment_info()),
              sidecar_path(output_file, "run_metadata"))
    print(f"Saved prediction to {output_file}")


if __name__ == "__main__":
    main()
